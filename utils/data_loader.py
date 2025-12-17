import logging
import os
import time
import threading
import datetime
from typing import Dict, Any

from . import db
from models.sentiment import FinBertSentimentAnalyzer, EmotionClassifier
from utils.news import NewsLoader
from config import (
    NEWS_COLLECTION_INTERVAL_SECONDS,
    MMI_UPDATE_INTERVAL_SECONDS,
    MIN_ARTICLE_LENGTH,
    MIN_SENTIMENT_CONFIDENCE,
)

_collector_thread = None
_collector_lock = threading.Lock()

class DataCollector:
    """
    Periodic data ingestion and MMI update orchestrator.

    Responsibilities:
    - Ingestion: fetch news data and write to DB
    - Preprocessing: text normalization, sentiment & emotion inference
    - Indexing: store enriched records into `sentiment_data`
    - Analytics: periodically recompute MMIs for all symbols
    """

    def __init__(self, db: db.DatabaseHandler, news_keywords):
        logging.info(
            'Initializing DataCollector with News sector keywords',
            extra={"pipeline_step": "collector_init"},
        )
        self.db = db
        self.news_loader = NewsLoader(keywords=news_keywords)
        self.sentiment = FinBertSentimentAnalyzer()
        self.emotion = EmotionClassifier()

        self.timing_config: Dict[str, Dict[str, Any]] = {
            'news_collection': {
                'interval_seconds': NEWS_COLLECTION_INTERVAL_SECONDS,
                'last_run': 0.0,
            },
            'mmi_update': {
                'interval_seconds': MMI_UPDATE_INTERVAL_SECONDS,
                'last_run': 0.0,
            },
        }

        self._stop_event = threading.Event()

        # Ensure sector news cache table exists (function is idempotent)
        if hasattr(self.db, '_ensure_sector_news_table'):
            self.db._ensure_sector_news_table()

    def log_daily_sentiment_distribution(self):
        import collections
        today = datetime.datetime.utcnow().date()
        rows = self.db.fetch_recent_entries(10000)
        # Only today's entries
        today_rows = [r for r in rows if str(today) in str(r[0])]
        labels = [r[7] for r in today_rows if r[7]]  # sentiment_label
        dist = collections.Counter(labels)
        logging.info(f"Sentiment distribution for {today}: {dict(dist)}")
        # Drift detection: compare with previous day
        prev_rows = [r for r in rows if str(today - datetime.timedelta(days=1)) in str(r[0])]
        prev_labels = [r[7] for r in prev_rows if r[7]]
        prev_dist = collections.Counter(prev_labels)
        if prev_dist:
            for label in dist:
                prev = prev_dist.get(label, 0)
                curr = dist[label]
                if prev > 0 and abs(curr - prev) / prev > 0.5:
                    logging.warning(f"Sentiment drift detected for '{label}': {prev} -> {curr}")
    def run_daily_metrics(self):
        self.log_daily_sentiment_distribution()

    def collection_loop(self, sector_map):
            """
            Optimized collection loop using Event.wait() instead of time.sleep().
            Benefits:
            - Immediate shutdown capability
            - Lower CPU usage (event-driven waiting)
            - More responsive to timing changes
            """
            logging.info(
                'Beginning continuous data collection loop with optimized scheduling',
                extra={"pipeline_step": "collector_loop"},
            )
            
            while not self._stop_event.is_set():
                current_time = time.time()
                
                # Calculate next wake time
                next_wake_times = []
                
                # Check and run news collection
                if self._should_run('news_collection', current_time):
                    logging.info(
                        'Running news collection cycle...',
                        extra={"pipeline_step": "news_collection"},
                    )
                    
                    try:
                        start_ts = time.time()
                        self.collect_sector_news_all(sector_map)
                        self.collect_news()
                        end_ts = time.time()
                        logging.info(
                            "News collection cycle completed",
                            extra={
                                "pipeline_step": "news_collection",
                                "duration_seconds": round(end_ts - start_ts, 2),
                            },
                        )
                        self.timing_config['news_collection']['last_run'] = current_time
                    except Exception as e:
                        logging.error(
                            f"Error in news collection: {e}",
                            extra={"error_category": "api"},
                        )
                
                # Calculate time until next news collection
                news_next = self._time_until_next_run('news_collection', current_time)
                next_wake_times.append(news_next)
                
                # Check and run MMI update
                if self._should_run('mmi_update', current_time):
                    logging.info(
                        'Running MMI update cycle...',
                        extra={"pipeline_step": "mmi_update"},
                    )
                    
                    try:
                        start_ts = time.time()
                        self.update_all_mmi()
                        end_ts = time.time()
                        logging.info(
                            'MMI update complete',
                            extra={
                                "pipeline_step": "mmi_update",
                                "duration_seconds": round(end_ts - start_ts, 2),
                            },
                        )
                        self.timing_config['mmi_update']['last_run'] = current_time
                    except Exception as e:
                        logging.error(
                            f"Error in MMI update: {e}",
                            extra={"error_category": "logic"},
                        )
                
                # Calculate time until next MMI update
                mmi_next = self._time_until_next_run('mmi_update', current_time)
                next_wake_times.append(mmi_next)
                
                # Sleep until the next task is due (whichever comes first)
                sleep_duration = min(next_wake_times)
                
                logging.debug(
                    f"Sleeping for {sleep_duration:.1f} seconds until next task",
                    extra={"pipeline_step": "collector_loop"},
                )
                
                if self._stop_event.wait(timeout=sleep_duration):
                    logging.info(
                        "Stop event received, exiting collection loop",
                        extra={"pipeline_step": "collector_loop"},
                    )
                    break
        
    def _should_run(self, task_name: str, current_time: float) -> bool:
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
    
    def _time_until_next_run(self, task_name: str, current_time: float) -> float:
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
    
    def stop(self) -> None:
        """
        Gracefully stop the collection loop.
        This will interrupt the wait and exit cleanly.
        """
        logging.info("Stopping data collector...")
        self._stop_event.set()
    
    def get_timing_status(self) -> Dict[str, Dict[str, str]]:
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
    
    def update_timing_config(self, task_name: str, new_interval_seconds: float) -> None:
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
                f'{old_interval}s → {new_interval_seconds}s',
                extra={"pipeline_step": "collector_config"},
            )
            # Interrupt current wait to apply changes immediately
            self._stop_event.set()
            self._stop_event.clear()
        else:
            logging.error(
                f'Unknown task: {task_name}',
                extra={"error_category": "logic"},
            )

    def collect_sector_news_all(self, sector_map) -> None:
        """Loop all sectors and cache fresh news (<24h). Fetch only if outdated or missing."""

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
                        logging.info(
                            f"Using cached sector news for {symbol} (fresh <24h, {len(recent_news)} stories)",
                            extra={"pipeline_step": "news_collection", "symbol": symbol},
                        )
                        continue
                else:
                    logging.info(
                        f"Cached sector news for {symbol} has unparsable timestamp ({latest_time}); refetching",
                        extra={"pipeline_step": "news_collection", "symbol": symbol},
                    )

            # 🚀 No fresh cache 
            logging.info(
                f"No fresh cache found for {symbol}, fetching from NewsData...",
                extra={"pipeline_step": "news_collection", "symbol": symbol},
            )
            articles = self.news_loader.fetch_raw_news(q=keyword)

            if articles:
                self.db.insert_sector_news(symbol, articles)
                logging.info(
                    f"Filled/updated sector news cache for {symbol} with {len(articles)}",
                    extra={
                        "pipeline_step": "news_collection",
                        "symbol": symbol,
                        "article_count": len(articles),
                    },
                )

    def collect_news(self, fetch_window_minutes=None) -> None:
        import hashlib
        from langdetect import detect, LangDetectException
        logging.info(
            'Fetching news articles for sectors...',
            extra={"pipeline_step": "news_collection"},
        )
        news = self.news_loader.fetch_news()
        seen_hashes = set()
        inserted = 0
        now = time.time()
        min_length = MIN_ARTICLE_LENGTH
        window_seconds = None
        if fetch_window_minutes is not None:
            window_seconds = fetch_window_minutes * 60
        finance_keywords = [
            'stock', 'market', 'nifty', 'sensex', 'shares', 'equity', 'ipo', 'dividend', 'earnings',
            'bull', 'bear', 'invest', 'finance', 'fii', 'dii', 'mutual fund', 'bond', 'index', 'sector',
            'bank', 'pharma', 'auto', 'fmcg', 'it', 'profit', 'loss', 'rally', 'crash', 'volatility', 'vix',
            'inflation', 'interest rate', 'reserve bank', 'rbi', 'nse', 'bse', 'exchange', 'futures', 'options',
            'derivative', 'portfolio', 'hedge', 'liquidity', 'capital', 'yield', 'valuation', 'macd', 'technical', 'fundamental'
        ]
        import re
        def normalize_text(text):
            # Lowercase
            text = text.lower()
            # Remove URLs
            text = re.sub(r'http\S+|www\.\S+', '', text)
            # Remove emojis (basic unicode ranges)
            text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
            # Remove tickers (e.g., $AAPL)
            text = re.sub(r'\$[a-zA-Z]+', '', text)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        for article in news:
            try:
                text = article['text']
                # Validate incoming data
                if not text or len(text.strip()) < min_length:
                    logging.info(
                        f"Skipped short/empty article for {article['symbol']} at {article['timestamp']}",
                        extra={"pipeline_step": "news_collection"},
                    )
                    continue
                # Finance-relevance filtering (simple keyword match)
                if not any(kw in text.lower() for kw in finance_keywords):
                    logging.info(
                        f"Skipped non-finance article for {article['symbol']} at {article['timestamp']}",
                        extra={"pipeline_step": "news_collection"},
                    )
                    continue
                # Language detection: skip non-English
                try:
                    lang = detect(text)
                    if lang != 'en':
                        logging.info(
                            f"Skipped non-English article for {article['symbol']} at {article['timestamp']}",
                            extra={"pipeline_step": "news_collection"},
                        )
                        continue
                except LangDetectException:
                    logging.info(
                        f"Skipped undetectable language for {article['symbol']} at {article['timestamp']}",
                        extra={"pipeline_step": "news_collection"},
                    )
                    continue
                # Configurable fetch window
                if window_seconds is not None:
                    try:
                        ts = article['timestamp']
                        ts_epoch = None
                        if isinstance(ts, (int, float)):
                            ts_epoch = float(ts)
                        else:
                            import dateutil.parser
                            ts_epoch = dateutil.parser.parse(str(ts)).timestamp()
                        if now - ts_epoch > window_seconds:
                            logging.info(
                                f"Skipped old article for {article['symbol']} at {article['timestamp']}",
                                extra={"pipeline_step": "news_collection"},
                            )
                            continue
                    except Exception as e:
                        logging.warning(
                            f"Could not parse timestamp for fetch window: {e}",
                            extra={"error_category": "logic"},
                        )
                        continue
                # Duplicate detection: hash of (text, source, timestamp)
                hash_input = (text + article['symbol'] + str(article['timestamp'])).encode('utf-8')
                content_hash = hashlib.sha256(hash_input).hexdigest()
                if content_hash in seen_hashes:
                    logging.info(
                        f"Duplicate detected for {article['symbol']} at {article['timestamp']}",
                        extra={"pipeline_step": "news_collection"},
                    )
                    continue
                seen_hashes.add(content_hash)
                cleaned_text = normalize_text(text)
                label, score, confidence = self.sentiment.predict_with_confidence(cleaned_text)
                emotion = self.emotion.predict(cleaned_text)
                if confidence < MIN_SENTIMENT_CONFIDENCE:
                    logging.info(
                        f"Low-confidence prediction for {article['symbol']} at {article['timestamp']}, skipping.",
                        extra={"pipeline_step": "news_collection"},
                    )
                    continue
                data = {
                    'timestamp': str(article['timestamp']),
                    'source': 'news',
                    'text': text,
                    'cleaned_text': cleaned_text,
                    'symbol': article['symbol'],
                    'sentiment_score': float(score),
                    'sentiment_label': label,
                    'sentiment_confidence': float(confidence),
                    'emotion': emotion,
                    'content_hash': content_hash,
                    'sector': article.get('sector', article['symbol']),
                    'ingestion_ts': time.time()
                }
                self.db.insert_entry(data)
                inserted += 1
            except Exception as e:
                logging.error(
                    f"Exception while processing article: {e}",
                    extra={"error_category": "logic"},
                )
        logging.info(
            f"Inserted {inserted} unique news articles into DB.",
            extra={"pipeline_step": "news_collection"},
        )
        
    def update_all_mmi(self):
        logging.info(
            'Updating all MMIs in the database...',
            extra={"pipeline_step": "mmi_update"},
        )
        all_rows = self.db.fetch_recent_entries(1000)
        # Filter out 'NEWS' symbol and get unique symbols
        symbols = set([row[4] if isinstance(row, (list, tuple)) else row['symbol'] for row in all_rows])
        symbols.discard('NEWS')  # Remove 'NEWS' symbol if present
        now = time.time()
        for sym in symbols:
            rec = self.db.get_mmi_for_topic(sym)
            if not rec or (now - rec['last_update']) >= 86400:
                df_topic = [row for row in all_rows if (row[4] if isinstance(row, (list, tuple)) else row['symbol'])==sym]
                self.db.recompute_and_store_mmi(sym, df_topic)
            else:
                logging.info(
                    f'Skipping MMI recompute for {sym}; cached value is fresh (<24h).',
                    extra={"pipeline_step": "mmi_update", "symbol": sym},
                )
        logging.info(
            'All MMIs updated (only outdated ones recomputed).',
            extra={"pipeline_step": "mmi_update"},
        )


def start_streams(sector_map, indices):
    """Start the background data collector exactly once per process.

    This is idempotent and safe to call multiple times (e.g., on Streamlit reloads).
    """
    global _collector_thread
    with _collector_lock:
        if _collector_thread and _collector_thread.is_alive():
            return _collector_thread

        def thread_target():
            # Create a fresh DB handler inside the thread to avoid sharing connections across threads
            db_thread = db.DatabaseHandler(indices)
            thread_collector = DataCollector(db=db_thread, news_keywords=list(sector_map.keys()))
            thread_collector.collection_loop(sector_map)

        t = threading.Thread(target=thread_target, daemon=True)
        t.start()
        _collector_thread = t
        return _collector_thread