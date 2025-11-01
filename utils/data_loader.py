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

    
        self.mmi_last_update = 0
        # Ensure sector news cache table exists (function is idempotent)
        if hasattr(self.db, '_ensure_sector_news_table'):
            self.db._ensure_sector_news_table()

    def collection_loop(self):
        logging.info('Beginning continuous data collection loop')
        while True:
            self.collect_sector_news_all()
            self.collect_news()  # Sentiment ingestion for legacy pipeline/compliance
            now = time.time()
            if now - self.mmi_last_update > 3600:
                logging.info('Triggering MMI updates for all topics')
                self.update_all_mmi()
                self.mmi_last_update = now
            time.sleep(180)  # collect every 3 minutes
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
                logging.error(f"Error processing news: {article.get('text','')} | Exception: {e}")
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
