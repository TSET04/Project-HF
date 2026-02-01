import os
import datetime
import logging
import json
import mysql.connector
import numpy as np
import yfinance as yf
import time
from textblob import TextBlob
import traceback
from utils.helper import helper
import threading

class DatabaseHandler:
    def __init__(self):
        mysql_host = os.environ.get('MYSQL_HOST')
        mysql_user = os.environ.get('MYSQL_USER')
        mysql_password = os.environ.get('MYSQL_PASSWORD')
        mysql_db = os.environ.get('MYSQL_DATABASE')
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            raise RuntimeError('MySQL configuration missing: set MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE')
        self.finbert_pipeline = None
        logging.info('Connecting to MySQL backend')
        
        # Store connection parameters for thread-local connections
        self.connection_params = {
            'host': mysql_host,
            'user': mysql_user,
            'password': mysql_password,
            'database': mysql_db,
            'autocommit': False
        }
        
        # Thread-local storage for connections
        self._local = threading.local()
        
        # Create initial connection for setup
        self.conn = mysql.connector.connect(**self.connection_params)
        
        # ensure MySQL tables exist
        self._ensure_table_mysql()
        self._ensure_mmi_table_mysql()
        self._ensure_sector_news_table()
        self._ensure_indices_table()
        self._ensure_trends_table()
        self._ensure_ai_feedback_table()
    
    def _get_connection(self):
        """Get thread-local connection or create new one"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            try:
                self._local.conn = mysql.connector.connect(**self.connection_params)
                logging.debug(f"Created new connection for thread {threading.current_thread().name}")
            except Exception as e:
                logging.error(f"Failed to create connection: {e}")
                raise
        return self._local.conn
    
    def _get_cursor(self, buffered=True):
        """Get a cursor with buffered=True by default to avoid unread results"""
        conn = self._get_connection()
        return conn.cursor(buffered=buffered)
        
    def _ensure_table_mysql(self):
        logging.info('Ensuring sentiment_data table existence (mysql)...')
        cursor = self.conn.cursor(buffered=True)
        try:
            cursor.execute('''CREATE TABLE IF NOT EXISTS sentiment_data (
                                id INT AUTO_INCREMENT PRIMARY KEY,
                                timestamp VARCHAR(255),
                                source VARCHAR(32),
                                text TEXT,
                                symbol VARCHAR(32),
                                sentiment_score FLOAT,
                                emotion VARCHAR(32))
                            ''')
            self.conn.commit()
        finally:
            cursor.close()
    
    def _ensure_mmi_table_mysql(self):
        logging.info('Ensuring mmi_cache table existence (mysql)...')
        cursor = self.conn.cursor(buffered=True)
        try:
            cursor.execute('''CREATE TABLE IF NOT EXISTS mmi_cache (
                                symbol VARCHAR(64) PRIMARY KEY,
                                mmi INT,
                                mood VARCHAR(64),
                                detail TEXT,
                                last_update DOUBLE)''')
            self.conn.commit()
            # Ensure existing column is large enough to hold mood strings we set
            try:
                cursor.execute("ALTER TABLE mmi_cache MODIFY mood VARCHAR(64)")
                self.conn.commit()
            except Exception as e:
                logging.debug(f"Could not alter mmi_cache.mood: {e}")
        finally:
            cursor.close()

    def _ensure_sector_news_table(self):
        cursor = self.conn.cursor(buffered=True)
        try:
            cursor.execute('''CREATE TABLE IF NOT EXISTS sector_news_cache (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(64),
                    news_title TEXT,
                    news_desc TEXT,
                    news_url VARCHAR(500),
                    published_at VARCHAR(64),
                    cached_at DOUBLE
                )''')
            self.conn.commit()
        finally:
            cursor.close()

    def _ensure_indices_table(self):
        cursor = self.conn.cursor(buffered=True)
        try:
            cursor.execute('''CREATE TABLE IF NOT EXISTS index_config (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    ticker VARCHAR(32) UNIQUE,
                    name VARCHAR(128),
                    active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
            self.conn.commit()
            
            # Seed default indices
            default_indices = [
                ('^NSEI', 'NIFTY 50'),
                ('^BSESN', 'BSE SENSEX'),
                ('^NSEBANK', 'NIFTY BANK'),
                ('^CNXIT', 'NIFTY IT'),
                ('^CNXFMCG', 'NIFTY FMCG')
            ]
            
            for ticker, name in default_indices:
                try:
                    cursor.execute(
                        "INSERT INTO index_config (ticker, name) VALUES (%s, %s) ON DUPLICATE KEY UPDATE name=%s",
                        (ticker, name, name)
                    )
                except Exception as e:
                    logging.debug(f"Index seed failed for {ticker}: {e}")
            self.conn.commit()
            
            cursor.execute('''CREATE TABLE IF NOT EXISTS indices_cache (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    ticker VARCHAR(32),
                    pct_change DOUBLE,
                    cached_at DOUBLE
                )''')
            # add index for faster lookup
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_indices_cache_ticker ON indices_cache(ticker)")
            except Exception:
                pass
            self.conn.commit()
        finally:
            cursor.close()

    def _ensure_trends_table(self):
        cursor = self.conn.cursor(buffered=True)
        try:
            cursor.execute('''CREATE TABLE IF NOT EXISTS trends_cache (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    term VARCHAR(255) UNIQUE,
                    value DOUBLE,
                    cached_at DOUBLE
                )''')
            # add index for term lookup
            try:
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_trends_cache_term ON trends_cache(term)")
            except Exception:
                pass
            self.conn.commit()
        finally:
            cursor.close()

    def _ensure_ai_feedback_table(self):
        cursor = self.conn.cursor(buffered=True)
        try:
            cursor.execute('''CREATE TABLE IF NOT EXISTS ai_feedback_cache (
                    symbol VARCHAR(64) PRIMARY KEY,
                    feedback TEXT,
                    cached_at DOUBLE
                )''')
            self.conn.commit()
        finally:
            cursor.close()

    def get_ai_feedback(self, symbol):
        cursor = self._get_cursor()
        try:
            cursor.execute("SELECT feedback, cached_at FROM ai_feedback_cache WHERE symbol=%s", (symbol,))
            row = cursor.fetchone()
            if not row:
                return None
            return {'feedback': row[0], 'cached_at': float(row[1])}
        except Exception as e:
            logging.warning(f"get_ai_feedback error for {symbol}: {e}")
            return None
        finally:
            cursor.close()

    def store_ai_feedback(self, symbol, feedback):
        now_ts = datetime.datetime.utcnow().timestamp()
        conn = self._get_connection()
        cursor = conn.cursor(buffered=True)
        try:
            existing = self.get_ai_feedback(symbol)
            if existing:
                cursor.execute("UPDATE ai_feedback_cache SET feedback=%s, cached_at=%s WHERE symbol=%s", (feedback, now_ts, symbol))
            else:
                cursor.execute("INSERT INTO ai_feedback_cache (symbol, feedback, cached_at) VALUES (%s, %s, %s)", (symbol, feedback, now_ts))
            conn.commit()
        except Exception as e:
            logging.warning(f"store_ai_feedback failed for {symbol}: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            cursor.close()

    def generate_and_store_ai_feedback(self, symbol, mmi_value: int):
        """
        Generate Mistral-based (or fallback) feedback for the given symbol and MMI value,
        store it in ai_feedback_cache and return the feedback string.

        Note: this code treats MMI range as -100..100 and mentions that in the prompt.
        """
        api_key = os.environ.get('MISTRAL_API_KEY')
        # deterministic label (based on -100..100)
        label = 'Neutral'
        if mmi_value <= -40:
            label = 'Bearish'
        elif mmi_value > 20:
            label = 'Bullish'

        # fallback deterministic message
        def fallback_msg(label, mmi_value):
            if label == 'Bullish':
                return f"Bullish — The MMI is {mmi_value}, indicating optimism in this sector. Consider investigating recent positive catalysts before buying."
            elif label == 'Bearish':
                return f"Bearish — The MMI is {mmi_value}, indicating fear in this sector. Exercise caution and review downside risks."
            else:
                return f"Neutral — The MMI is {mmi_value}, market mood appears balanced; wait for clearer signals."

        if not api_key:
            logging.warning("MISTRAL_API_KEY not found, using fallback AI feedback.")
            fb = fallback_msg(label, mmi_value)
            try:
                self.store_ai_feedback(symbol, fb)
            except Exception as e:
                logging.debug(f"Failed to store fallback ai feedback: {e}")
            return fb

        # call Mistral
        try:
            prompt = (
            "You are an unbiased financial assistant.\n\n"
            "Context:\n"
            "The Market Mood Index (MMI) is a composite sentiment indicator that summarizes overall market psychology "
            "using multiple signals such as trend strength, volatility, participation, momentum, and risk appetite. "
            "It does NOT predict price, but reflects whether the market is emotionally overheated, fearful, or balanced.\n\n"
            "Interpretation rules:\n"
            "- Low MMI implies fear, risk-off behavior, and bearish market mood.\n"
            "- Mid-range MMI implies balance, uncertainty, or neutral market mood.\n"
            "- High MMI implies optimism, risk-on behavior, and bullish market mood.\n\n"
            "Task:\n"
            f"Given only the current MMI value ({mmi_value}) on a scale from 0 to 100, "
            "provide factual, sentiment-based guidance on whether the user should buy, sell, or stay calm. "
            "Also classify the market mood as bullish, bearish, or neutral based purely on sentiment interpretation.\n\n"
            "Constraints:\n"
            "- Do NOT repeat the MMI value or explain the scale.\n"
            "- Do NOT make price predictions or guarantees.\n"
            "- Keep the response concise, neutral, and easy to understand.\n"
            "- Avoid numbers in square brackets.\n"
            )


            url = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            data = {
                "model": "mistral-large-latest", 
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 120,
            }

            logging.info(f"Calling Mistral for symbol={symbol} mmi={mmi_value}")
            session = helper.get_retry_session(retries=2, backoff_factor=1.5)
            resp = session.post(url, headers=headers, json=data, timeout=15)
            resp.raise_for_status()

            try:
                result = resp.json()
            except ValueError:
                logging.warning("Mistral returned non-JSON response")
                raise RuntimeError("Invalid JSON from Mistral")
            logging.debug(f"Mistral response raw: {result}")
            result_text = helper.parse_mistral_response(result) if hasattr(helper, "parse_mistral_response") else str(result)

            # Clean up response
            lines = [l.strip() for l in result_text.split("\n") if l.strip()]
            if lines:
                fb = " ".join(lines[:2])
            else:
                fb = result_text

            # Additional cleanup: remove markdown, citations, etc.
            fb = fb.replace("**", "").replace("[", "").replace("]", "")
            fb = ' '.join(fb.split())
            logging.info(f"Mistral produced feedback for {symbol}: {fb}")

        except Exception as e:
            logging.warning(f'Mistral API call failed for {symbol}: {e}')
            logging.debug(traceback.format_exc())
            fb = fallback_msg(label, mmi_value)

        try:
            self.store_ai_feedback(symbol, fb)
            logging.info(f'Stored AI feedback for {symbol} (mmi={mmi_value})')
        except Exception as e:
            logging.warning(f'Failed to store AI feedback for {symbol}: {e}')
        return fb
        
    def insert_sector_news(self, symbol, articles):
        """
        Expects: articles = [{title, description, url, publishedAt}, ...]
        Prunes previous entries for symbol, inserts up to 10 new ones.
        """
        conn = self._get_connection()
        cursor = conn.cursor(buffered=True)
        try:
            # Prune any old stories for this symbol
            cursor.execute("DELETE FROM sector_news_cache WHERE symbol=%s", (symbol,))
            max_cache_time = time.time()
            # Insert up to 10 latest
            for a in articles[:10]:
                cursor.execute(
                    "INSERT INTO sector_news_cache (symbol, news_title, news_desc, news_url, published_at, cached_at) VALUES (%s, %s, %s, %s, %s, %s)",
                    (symbol, a.get('title'), a.get('description'), a.get('url'), a.get('publishedAt'), max_cache_time)
                )
            conn.commit()
        except Exception as e:
            logging.warning(f"insert_sector_news failed for {symbol}: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            cursor.close()

    def get_sector_news(self, symbol):
        """
        Returns up to 10 cached stories for symbol (fresh only <24h)
        """
        cursor = self._get_cursor()
        try:
            min_time = time.time() - 86400
            cursor.execute(
                "SELECT news_title, news_desc, news_url, published_at FROM sector_news_cache WHERE symbol=%s AND cached_at>=%s ORDER BY published_at DESC LIMIT 10",
                (symbol, min_time))
            results = cursor.fetchall()
            return results
        except Exception as e:
            logging.warning(f"get_sector_news error for {symbol}: {e}")
            return []
        finally:
            cursor.close()
    

    def get_mmi_for_topic(self, symbol):
        logging.info(f'Fetching MMI for topic: {symbol}')
        cursor = self._get_cursor()
        try:
            cursor.execute("SELECT mmi, mood, detail, last_update FROM mmi_cache WHERE symbol=%s", (symbol,))
            row = cursor.fetchone()
            if not row:
                return None
            return {'mmi': int(row[0]), 'mood': row[1], 'detail': row[2], 'last_update': float(row[3]) if row[3] is not None else None}
        except Exception as e:
            logging.warning(f"get_mmi_for_topic failed for {symbol}: {e}")
            return None
        finally:
            cursor.close()

    def recompute_and_store_mmi(self, symbol, rows, user_feedbacks=None):
        """
        Recompute and store MMI for a symbol.
        - rows: list of recent sentiment_data rows (tuples or dicts). We treat it read-only; we do not overwrite it.
        """
        logging.info(f'Recomputing/storing MMI for topic: {symbol}')
        conn = self._get_connection()
        now_ts = datetime.datetime.utcnow().timestamp()

        try:
            # Re-check cached MMI freshness
            # Skip computation if fresh cache exists
            existing = self.get_mmi_for_topic(symbol)
            one_day_ago = now_ts - 86400
            if existing and existing.get('last_update') and float(existing.get('last_update')) >= one_day_ago:
                logging.info(f'Using cached MMI for {symbol} (age <24h)')
                return existing['mmi']

            # -------------------------
            # Fetch active index tickers from database (use different variable name so we don't shadow `rows`)
            idx_cursor = conn.cursor(buffered=True)
            try:
                idx_cursor.execute("SELECT ticker FROM index_config WHERE active=TRUE")
                index_rows = idx_cursor.fetchall()
                index_tickers = [r[0] for r in index_rows]
            except Exception as e:
                logging.warning(f"Failed to read index_config: {e}")
                index_tickers = []
            finally:
                idx_cursor.close()

            if not index_tickers:
                # Fallback to defaults if table is empty
                index_tickers = ['^NSEI', '^BSESN', '^NSEBANK', '^CNXIT', '^CNXFMCG']

            index_changes = []
            index_map = {}
            fresh_map = {}

            # try to get cached index values
            idx_cache_cursor = conn.cursor(buffered=True)
            try:
                placeholders = ','.join(['%s'] * len(index_tickers))
                q = f"SELECT ticker, pct_change, cached_at FROM indices_cache WHERE ticker IN ({placeholders})"
                params = tuple(index_tickers)
                idx_cache_cursor.execute(q, params)
                rows_idx = idx_cache_cursor.fetchall()
                for r in rows_idx:
                    try:
                        ticker = r[0]
                        pct = float(r[1]) if r[1] is not None else None
                        cached_at = float(r[2]) if r[2] is not None else 0.0
                        if pct is not None:
                            fresh_map[ticker] = (pct, cached_at)
                            if cached_at >= one_day_ago:
                                index_changes.append(pct)
                    except Exception:
                        continue
            except Exception as e:
                logging.warning(f"indices_cache lookup failed: {e}")
            finally:
                idx_cache_cursor.close()

            # fetch missing tickers from yfinance and store them with cached_at
            missing = [t for t in index_tickers if t not in fresh_map]
            if missing:
                ins_cursor = conn.cursor(buffered=True)
                try:
                    for code in missing:
                        try:
                            data = yf.download(code, period='2d', interval='1d', progress=False, auto_adjust=False)
                            if data is not None and not data.empty and len(data['Close'].values) >= 2:
                                pct = (data['Close'].values[-1] - data['Close'].values[-2]) / float(data['Close'].values[-2])
                                index_changes.append(float(pct))
                                index_map[code] = float(pct)
                                # insert into cache (try update then insert to avoid duplicate issues)
                                try:
                                    ins_cursor.execute("INSERT INTO indices_cache (ticker, pct_change, cached_at) VALUES (%s, %s, %s)",
                                                       (code, float(pct), now_ts))
                                except Exception:
                                    try:
                                        ins_cursor.execute("UPDATE indices_cache SET pct_change=%s, cached_at=%s WHERE ticker=%s",
                                                           (float(pct), now_ts, code))
                                    except Exception as e:
                                        logging.debug(f"Failed to upsert indices_cache for {code}: {e}")
                        except Exception as e:
                            logging.debug(f"yfinance fetch failed for {code}: {e}")
                            continue
                    try:
                        conn.commit()
                    except Exception:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                finally:
                    ins_cursor.close()

            # include any cached values we loaded earlier in the index_map if not present
            for t in index_tickers:
                if t in fresh_map and t not in index_map:
                    index_map[t] = float(fresh_map[t][0])

            # normalize index changes to 0..1 range; guard empty lists
            if index_changes:
                # using a fixed window but guard extreme outliers by clipping then normalizing
                min_v, max_v = -0.05, 0.05
                normed = [max(0.0, min(1.0, (c - min_v) / (max_v - min_v))) for c in index_changes]
                index_perf_score = float(np.mean(normed)) if normed else 0.5
            else:
                index_perf_score = 0.5

            # --- News: prefer cached sector_news_cache (<24h) else use rows passed in (sentiment_data)
            news_texts = []
            cached_news = self.get_sector_news(symbol)
            if cached_news:
                for t, d, url, pub in cached_news:
                    txt = (t or '') + ' ' + (d or '')
                    if txt.strip():
                        news_texts.append(txt.strip())
                logging.info(f"Using {len(news_texts)} cached news items for {symbol}")

            # If no cached news, try to extract from provided 'rows' (sentiment_data)
            if not news_texts and rows:
                logging.info(f"No cached news found for {symbol}, checking provided sentiment_data rows")
                for r in rows:
                    try:
                        # support both tuple and dict rows
                        src = r[2] if isinstance(r, (list, tuple)) else r.get('source')
                        if src == 'news':
                            row_symbol = r[4] if isinstance(r, (list, tuple)) else r.get('symbol')
                            if row_symbol == symbol or symbol == 'market':
                                text = r[3] if isinstance(r, (list, tuple)) else r.get('text')
                                if text and len(text.strip()) > 20:
                                    news_texts.append(text.strip())
                    except Exception as e:
                        logging.debug(f"Error processing provided row: {e}")
                        continue
                logging.info(f"Found {len(news_texts)} news items from provided sentiment_data for {symbol}")

            if not news_texts:
                logging.warning(
                    f"No news found for {symbol}. "
                    f"The data_loader should have populated sector_news_cache. "
                    f"MMI will be computed with limited data."
                )

            # --- Clean and deduplicate texts ---
            news_texts = helper.deduplicate_texts(news_texts, min_length=20) if news_texts else []
            logging.info(f"After deduplication: {len(news_texts)} unique news items")

            # --- FinBERT Sentiment Analysis ---
            news_scores = []
            if news_texts:
                try:
                    from transformers import pipeline
                    if self.finbert_pipeline is None:
                        try:
                            self.finbert_pipeline = helper._load_finbert_model()
                            logging.info("FinBERT model loaded successfully.")
                        except Exception as e:
                            logging.warning(f"FinBERT load failed, using default sentiment model: {e}")
                            self.finbert_pipeline = pipeline('sentiment-analysis')

                    finbert = self.finbert_pipeline

                    for txt in news_texts:
                        try:
                            out = finbert(txt[:1000])[0]
                            lbl = out.get('label', '').lower()
                            score = float(out.get('score', 0.0))

                            if 'neg' in lbl:
                                val = -score
                            elif 'pos' in lbl:
                                val = score
                            else:
                                val = 0.0

                            news_scores.append(val)

                        except Exception as e:
                            logging.warning(f"FinBERT failed on text fragment: {e}")
                            continue

                except Exception as e:
                    # fallback to sentiment_score present in rows if available
                    logging.warning(f"FinBERT pipeline unavailable, falling back to stored sentiment_score: {e}")
                    for r in rows or []:
                        try:
                            src = r[2] if isinstance(r, (list, tuple)) else r.get('source')
                            if src == 'news':
                                score = r[5] if isinstance(r, (list, tuple)) else r.get('sentiment_score')
                                if score is not None:
                                    news_scores.append(float(score))
                        except Exception:
                            continue

            # --- Compute final sentiment ---
            news_sentiment = float(np.mean(news_scores)) if news_scores else 0.0
            news_sentiment = max(-1.0, min(1.0, news_sentiment))
            logging.info(f"Final news_sentiment for {symbol}: {news_sentiment:.3f}")

            # --- Trends: prefer cached trends_cache (<24h) else fetch and store via NewsData.io ---
            bull_terms = ["nifty today", "buy stocks", "stock market india", "buy shares", "RBI interest rates", "long term investment india"]
            bear_terms = ["sensex crash", "sell stocks", "stock market crash", "inflation india", "sell shares"]
            trend_sentiment = 0.0

            try:
                terms = bull_terms + bear_terms
                values = {}

                # 1) Delete stale cache (>24h)
                trends_cursor = conn.cursor(buffered=True)
                try:
                    trends_cursor.execute("DELETE FROM trends_cache WHERE cached_at < %s", (one_day_ago,))
                    conn.commit()
                except Exception as e:
                    logging.debug(f"Failed to prune trends_cache: {e}")

                # 2) Check cache
                try:
                    trends_cursor.execute("SELECT COUNT(*) FROM trends_cache")
                    count = trends_cursor.fetchone()[0]
                    table_empty = (count == 0)
                except Exception as e:
                    logging.debug(f"Failed to check trends_cache count: {e}")
                    table_empty = True

                if not table_empty:
                    for term in terms:
                        try:
                            trends_cursor.execute(
                                "SELECT value, cached_at FROM trends_cache WHERE term=%s AND value IS NOT NULL",
                                (term,),
                            )
                            rowt = trends_cursor.fetchone()
                            if rowt and float(rowt[1]) >= one_day_ago:
                                values[term] = float(rowt[0])
                                logging.info(f"Used stored trend value for {term}: {values[term]}")
                        except Exception as e:
                            logging.debug(f"Trend cache lookup failed for {term}: {e}")
                            continue
                else:
                    logging.info("trends_cache is empty or unavailable, fetching fresh data...")

                # 3) Fetch missing terms via NewsData.io sentiment
                missing_terms = [t for t in terms if t not in values]
                if missing_terms:
                    api_key = os.getenv("NEWSDATA_API_KEY")
                    if not api_key:
                        logging.warning("Missing NEWSDATA_API_KEY environment variable; skipping trend fetch.")
                        missing_terms = []
                    else:
                        base_url = "https://newsdata.io/api/1/news"
                        now_ts_local = datetime.datetime.utcnow().timestamp()

                        for kw in missing_terms:
                            try:
                                logging.info(f"Fetching news sentiment for: {kw}")
                                # build params safe (avoid direct string concatenation)
                                params = {
                                    "apikey": api_key,
                                    "q": kw,
                                    "country": "in",
                                    "language": "en"
                                }
                                session = helper.get_retry_session(retries=3, backoff_factor=2)
                                resp = session.get(base_url, params=params, timeout=20)
                                try:
                                    data = resp.json()
                                except ValueError:
                                    logging.warning(f"Invalid JSON from NewsData for keyword {kw}")
                                    continue

                                articles = data.get("results", [])
                                if not articles:
                                    logging.info(f"No articles found for keyword: {kw}")
                                    continue

                                scores = []
                                for art in articles:
                                    text = (art.get("title") or "") + " " + (art.get("description") or "")
                                    if text.strip():
                                        polarity = TextBlob(text).sentiment.polarity
                                        scores.append(polarity)

                                avg_score = float(np.mean(scores)) if scores else 0.0
                                # keep as float precision
                                normalized_val = (avg_score + 1.0) * 50.0  # -1..1 → 0..100 float

                                try:
                                    # upsert semantics: REPLACE or INSERT/UPDATE
                                    trends_cursor.execute(
                                        "REPLACE INTO trends_cache (term, value, cached_at) VALUES (%s, %s, %s)",
                                        (kw, normalized_val, now_ts_local)
                                    )
                                    conn.commit()
                                except Exception as e:
                                    logging.debug(f"Failed to write trend cache for {kw}: {e}")
                                    try:
                                        conn.rollback()
                                    except Exception:
                                        pass

                                values[kw] = normalized_val

                            except Exception as e:
                                logging.warning(f"Error fetching news trend for {kw}: {e}")
                                logging.debug(traceback.format_exc())

                            time.sleep(1)  # keep small delay to be polite with API

                # 4) Compute sentiment from values we have
                bull_vals = [values[k] / 100.0 for k in bull_terms if k in values]
                bear_vals = [values[k] / 100.0 for k in bear_terms if k in values]
                bull_avg = float(np.mean(bull_vals)) if bull_vals else 0.0
                bear_avg = float(np.mean(bear_vals)) if bear_vals else 0.0
                trend_sentiment = bull_avg - bear_avg
                trend_sentiment = max(-1.0, min(1.0, trend_sentiment))
            except Exception as e:
                logging.warning(f"NewsData.io trend sentiment failed: {e}")
                logging.debug(traceback.format_exc())
                trend_sentiment = 0.0
            finally:
                try:
                    trends_cursor.close()
                except Exception:
                    pass

            # user feedback aggregation
            feedback_sentiment = None
            if user_feedbacks:
                try:
                    cleaned = [max(-1.0, min(1.0, float(x))) for x in user_feedbacks]
                    feedback_sentiment = float(np.mean(cleaned)) if cleaned else None
                except Exception as e:
                    logging.debug(f"user_feedbacks processing failed: {e}")
                    feedback_sentiment = None

            if feedback_sentiment is not None:
                user_feedback_score = 0.7 * trend_sentiment + 0.3 * feedback_sentiment
            else:
                user_feedback_score = trend_sentiment
            user_feedback_score = max(-1.0, min(1.0, user_feedback_score))

            # Final MMI calculation (raw values -1..1)
            final_value = 0.4 * news_sentiment + 0.3 * user_feedback_score + 0.3 * index_perf_score
            final_value_clipped = max(-1.0, min(1.0, final_value))
            # Scale to -100..100 integer
            scaled_mmi = int(final_value_clipped * 100)
            scaled_mmi = max(-100, min(100, scaled_mmi))

            if scaled_mmi <= -40:
                mood = "Extreme Fear"
            elif scaled_mmi <= 20:
                mood = "Cautious / Neutral"
            else:
                mood = "Optimistic / Greedy"

            detail = json.dumps({
                'news_sentiment': float(news_sentiment),
                'trend_sentiment': float(trend_sentiment),
                'user_feedback_sentiment': float(user_feedback_score),
                'index_perf_score': float(index_perf_score),
                'index_changes': [float(x) for x in index_changes],
                'index_map': {k: float(v) for k, v in index_map.items()},
                'weights': {'news': 0.4, 'user_feedback': 0.3, 'indices': 0.3},
                'final_value_raw': float(final_value),
                'final_value_clipped': float(final_value_clipped),
                'scaled_mmi': scaled_mmi,
                'symbol': symbol
            })
            last_update = now_ts

            # commit MMI to DB
            mmi_cursor = conn.cursor(buffered=True)
            try:
                if self.get_mmi_for_topic(symbol):
                    mmi_cursor.execute("UPDATE mmi_cache SET mmi=%s, mood=%s, detail=%s, last_update=%s WHERE symbol=%s",
                                   (scaled_mmi, mood, detail, last_update, symbol))
                else:
                    mmi_cursor.execute(
                        "INSERT INTO mmi_cache (symbol, mmi, mood, detail, last_update) VALUES (%s, %s, %s, %s, %s)",
                        (symbol, scaled_mmi, mood, detail, last_update))
                conn.commit()
            except Exception as e:
                logging.warning(f"Failed to write mmi_cache for {symbol}: {e}")
                try:
                    conn.rollback()
                except Exception:
                    pass
            finally:
                mmi_cursor.close()

            return scaled_mmi

        except Exception as e:
            logging.error(f"recompute_and_store_mmi failed for {symbol}: {e}")
            logging.debug(traceback.format_exc())
            # Return existing or 0 on failure
            existing = self.get_mmi_for_topic(symbol)
            return existing['mmi'] if existing else 0

    def insert_entry(self, data):
        conn = self._get_connection()
        cursor = conn.cursor(buffered=True)
        try:
            cursor.execute(
                "INSERT INTO sentiment_data (timestamp, source, text, symbol, sentiment_score, emotion) VALUES (%s, %s, %s, %s, %s, %s)",
                (data['timestamp'], data['source'], data['text'], data['symbol'], data['sentiment_score'], data['emotion'])
            )
            conn.commit()
        except Exception as e:
            logging.warning(f"insert_entry failed: {e}")
            try:
                conn.rollback()
            except Exception:
                pass
        finally:
            cursor.close()

    def fetch_recent_entries(self, limit=100):
        cursor = self._get_cursor()
        try:
            cursor.execute("SELECT * FROM sentiment_data ORDER BY timestamp DESC LIMIT %s", (limit,))
            rows = cursor.fetchall()
            return rows
        finally:
            cursor.close()