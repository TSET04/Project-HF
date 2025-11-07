import logging
import os
import time
import threading
import datetime
from . import db
from models.sentiment import FinBertSentimentAnalyzer, EmotionClassifier
from utils.news import NewsLoader

_collector_thread = None
_collector_lock = threading.Lock()

class DataCollector:
    def __init__(self, db, news_keywords=None):
        logging.info('Initializing DataCollector with News sector keywords')
        self.db = db
        self.news_loader = NewsLoader(keywords=(news_keywords or [
            "banking sector india", "it sector india", "pharma sector india", "fmcg sector india", "auto sector india"
        ]))
        self.sentiment = FinBertSentimentAnalyzer()
        self.emotion = EmotionClassifier()

        self.timing_config = {
            'news_collection': {
                'interval_seconds': 21600,      # 6 hours
                'last_run': 0,
            },
            'mmi_update': {
                'interval_seconds': 86400,     # 24 hours
                'last_run': 0,
            }
        }        

        self._stop_event = threading.Event()
        self._wakeup_event = threading.Event()

        # Ensure sector news cache table exists (function is idempotent)
        if hasattr(self.db, '_ensure_sector_news_table'):
            self.db._ensure_sector_news_table()

    def collection_loop(self):
            """
            Optimized collection loop using Event.wait() instead of time.sleep().
            Benefits:
            - Immediate shutdown capability
            - Lower CPU usage (event-driven waiting)
            - More responsive to timing changes
            """
            logging.info('Beginning continuous data collection loop with optimized scheduling')
            
            while not self._stop_event.is_set():
                current_time = time.time()
                
                # Calculate next wake time
                next_wake_times = []
                
                # Check and run news collection
                if self._should_run('news_collection', current_time):
                    logging.info('Running news collection cycle...')
                    
                    try:
                        self.collect_sector_news_all()
                        self.collect_news()
                        self.timing_config['news_collection']['last_run'] = current_time
                        logging.info('News collection complete')
                    except Exception as e:
                        logging.error(f"Error in news collection: {e}")
                
                # Calculate time until next news collection
                news_next = self._time_until_next_run('news_collection', current_time)
                next_wake_times.append(news_next)
                
                # Check and run MMI update
                if self._should_run('mmi_update', current_time):
                    logging.info('Running MMI update cycle...')
                    
                    try:
                        self.update_all_mmi()
                        self.timing_config['mmi_update']['last_run'] = current_time
                        logging.info('MMI update complete. Next run in 1 hour.')
                    except Exception as e:
                        logging.error(f"Error in MMI update: {e}")
                
                # Calculate time until next MMI update
                mmi_next = self._time_until_next_run('mmi_update', current_time)
                next_wake_times.append(mmi_next)
                
                # Sleep until the next task is due (whichever comes first)
                sleep_duration = min(next_wake_times)
                
                logging.debug(f"Sleeping for {sleep_duration:.1f} seconds until next task")
                
                if self._stop_event.wait(timeout=sleep_duration):
                    logging.info("Stop event received, exiting collection loop")
                    break
        
    def _should_run(self, task_name, current_time):
        """
        Check if enough time has passed since last run.
        
        Args:
            task_name: Key from timing_config
            current_time: Current timestamp
            
        Returns:
            bool: True if task should run
        """
        task = self.timing_config[task_name]
        time_elapsed = current_time - task['last_run']
        return time_elapsed >= task['interval_seconds']
    
    def _time_until_next_run(self, task_name, current_time):
        """
        Calculate how many seconds until this task should run again.
        
        Args:
            task_name: Key from timing_config
            current_time: Current timestamp
            
        Returns:
            float: Seconds until next run (minimum 0)
        """
        task = self.timing_config[task_name]
        time_elapsed = current_time - task['last_run']
        time_remaining = task['interval_seconds'] - time_elapsed
        return max(0, time_remaining)
    
    def stop(self):
        """
        Gracefully stop the collection loop.
        This will interrupt the wait and exit cleanly.
        """
        logging.info("Stopping data collector...")
        self._stop_event.set()
    
    def get_timing_status(self):
        """
        Get human-readable status of all timed tasks.
        Useful for debugging and monitoring.
        """
        current_time = time.time()
        status = {}
        
        for task_name, config in self.timing_config.items():
            time_since_last = current_time - config['last_run']
            time_until_next = self._time_until_next_run(task_name, current_time)
            
            status[task_name] = {
                'interval': f"{config['interval_seconds'] / 60} minutes",
                'last_run_ago': f"{time_since_last / 60:.1f} minutes ago",
                'next_run_in': f"{time_until_next / 60:.1f} minutes",
                'is_due': self._should_run(task_name, current_time)
            }
        
        return status
    
    def update_timing_config(self, task_name, new_interval_seconds):
        """
        Dynamically adjust timing intervals without restarting.
        The change will take effect on the next loop iteration.
        
        Args:
            task_name: 'news_collection' or 'mmi_update'
            new_interval_seconds: New interval in seconds
        """
        if task_name in self.timing_config:
            old_interval = self.timing_config[task_name]['interval_seconds']
            self.timing_config[task_name]['interval_seconds'] = new_interval_seconds
            logging.info(
                f'Updated {task_name} interval: '
                f'{old_interval}s → {new_interval_seconds}s'
            )
            # Interrupt current wait to apply changes immediately
            self._stop_event.set()
            self._stop_event.clear()
        else:
            logging.error(f'Unknown task: {task_name}')

    def collect_sector_news_all(self):
        """Loop all sectors and cache fresh news (<24h). Fetch only if outdated or missing."""
        sector_map = getattr(self, 'sector_map', {
            'banking': 'banking sector india',
            'information technology': 'it sector india',
            'pharma': 'pharma sector india',
            'fmcg': 'fmcg sector india',
            'auto': 'auto sector india'
        })

        def _parse_timestamp(ts):
            if not ts:
                return None
            # handle numeric epoch
            try:
                if isinstance(ts, (int, float)):
                    return datetime.datetime.fromtimestamp(float(ts))
            except Exception:
                pass
            s = str(ts)
            # ISO with Z -> convert to offset
            try:
                if s.endswith('Z'):
                    return datetime.datetime.fromisoformat(s.replace('Z', '+00:00')).replace(tzinfo=None)
            except Exception:
                pass
            # try fromisoformat for common ISO and space-separated
            try:
                return datetime.datetime.fromisoformat(s)
            except Exception:
                pass
            # try common strptime patterns
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.datetime.strptime(s, fmt)
                except Exception:
                    continue
            # fallback: give up
            logging.warning(f"Unable to parse timestamp: {s}")
            return None

        for symbol, keyword in sector_map.items():
            # Fetch the most recent news for this sector
            recent_news = self.db.get_sector_news(symbol)

            # Check if recent cache (<24h) exists
            if recent_news and len(recent_news) >= 5:
                latest_time = recent_news[0][3] if isinstance(recent_news[0], (list, tuple)) else recent_news[0].get('publishedAt')
                dt = _parse_timestamp(latest_time)
                if dt is not None:
                    age_hours = (time.time() - dt.timestamp()) / 3600
                    if age_hours < 24:
                        logging.info(f"Using cached sector news for {symbol} (fresh <24h, {len(recent_news)} stories)")
                        continue
                else:
                    logging.info(f"Cached sector news for {symbol} has unparsable timestamp ({latest_time}); refetching")

            # 🚀 No fresh cache 
            logging.info(f"No fresh cache found for {symbol}, fetching from NewsData...")
            articles = self.news_loader.fetch_raw_news(q=keyword, count=10)

            if articles:
                self.db.insert_sector_news(symbol, articles)
                logging.info(f"Filled/updated sector news cache for {symbol} with {len(articles)}")

    def collect_news(self):
        logging.info('Fetching news articles for sectors...')
        news = self.news_loader.fetch_news(count=10)
        for article in news:
            try:
                score = self.sentiment.predict(article['text'])
                emotion = self.emotion.predict(article['text'])
                data = {
                    'timestamp': str(article['timestamp']),
                    'source': 'news',
                    'text': article['text'],
                    'symbol': article['symbol'],
                    'sentiment_score': score,
                    'emotion': emotion
                }
                self.db.insert_entry(data)
            except Exception as e:
                logging.error(f"Exception: {e}")
        logging.info(f"Inserted {len(news)} news articles into DB.")
        
    def update_all_mmi(self):
        logging.info('Updating all MMIs in the database...')
        all_rows = self.db.fetch_recent_entries(1000)
        symbols = set([row[4] if isinstance(row, (list, tuple)) else row['symbol'] for row in all_rows])
        now = time.time()
        for sym in symbols:
            rec = self.db.get_mmi_for_topic(sym)
            if not rec or (now - rec['last_update']) >= 86400:
                df_topic = [row for row in all_rows if (row[4] if isinstance(row, (list, tuple)) else row['symbol'])==sym]
                self.db.recompute_and_store_mmi(sym, df_topic)
            else:
                logging.info(f'Skipping MMI recompute for {sym}; cached value is fresh (<24h).')
        logging.info('All MMIs updated (only outdated ones recomputed).')


def start_streams(db_config=None):
    """Start the background data collector exactly once per process.

    This is idempotent and safe to call multiple times (e.g., on Streamlit reloads).
    """
    global _collector_thread
    with _collector_lock:
        if _collector_thread and _collector_thread.is_alive():
            return _collector_thread

        def thread_target():
            # Create a fresh DB handler inside the thread to avoid sharing connections across threads
            db_thread = db.DatabaseHandler()
            thread_collector = DataCollector(db_thread)
            thread_collector.collection_loop()

        t = threading.Thread(target=thread_target, daemon=True)
        t.start()
        _collector_thread = t
        return _collector_thread