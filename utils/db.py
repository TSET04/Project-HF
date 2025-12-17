import os
import datetime
import logging
import json
import mysql.connector
import numpy as np
import time
import traceback
from utils.helper import helper
import threading

class DatabaseHandler:
    def __init__(self, indices):
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
        self._ensure_indices_table(indices)
        self._ensure_trends_table()
        self._ensure_ai_feedback_table()
        self._ensure_mmi_history_table()
    
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
            # Ensure optional explainability column exists
            try:
                cursor.execute("ALTER TABLE mmi_cache ADD COLUMN news_summary TEXT")
                self.conn.commit()
            except Exception:
                # Column already exists or cannot be altered – safe to ignore
                pass
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
                    news_url VARCHAR(1000),
                    published_at VARCHAR(64),
                    cached_at DOUBLE
                )''')
            self.conn.commit()
        finally:
            cursor.close()

    def _ensure_indices_table(self, indices):
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
            
            for ticker, name in indices:
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

    def _ensure_mmi_history_table(self):
        """Table to store daily MMI history snapshots for each symbol."""
        cursor = self.conn.cursor(buffered=True)
        try:
            cursor.execute(
                '''CREATE TABLE IF NOT EXISTS mmi_history (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        symbol VARCHAR(64),
                        mmi INT,
                        as_of DOUBLE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY uniq_symbol_asof (symbol, as_of)
                    )'''
            )
            try:
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_mmi_history_symbol_asof ON mmi_history(symbol, as_of)"
                )
            except Exception:
                pass
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
        Generate Perplexity-based (or fallback) feedback for the given symbol and MMI value,
        store it in ai_feedback_cache and return the feedback string.

        Note: this code treats MMI range as -100..100 and mentions that in the prompt.
        """
        # Skip generating feedback for "NEWS" symbol
        if symbol == "NEWS":
            return None
        api_key = os.environ.get('MISTRAL_API_KEY')
        # deterministic label (based on -100..100)
        label = 'Neutral'
        if mmi_value <= 25:
            label = "Extreme Fear"
        elif mmi_value <= 50:
            label = "Fearful / Cautious"
        elif mmi_value <= 75:
            label = "Neutral to Optimistic"
        else:
            label = "Extreme Greed / Euphoria"

        # fallback deterministic message
        def fallback_msg(label, mmi_value):
            if label == 'Extreme Fear':
                return f"Extreme Fear — The MMI is {mmi_value}, indicating panic and strong risk aversion. Markets may be oversold; contrarian opportunities could exist, but caution is advised."
            elif label == 'Fearful / Cautious':
                return f"Fearful / Cautious — The MMI is {mmi_value}, showing that sentiment is still defensive. Consider waiting for stronger confirmation before taking major positions."
            elif label == 'Neutral to Optimistic':
                return f"Neutral to Optimistic — The MMI is {mmi_value}, suggesting improving confidence and moderate bullish bias. Gradual accumulation may be reasonable."
            else:
                return f"Extreme Greed / Euphoria — The MMI is {mmi_value}, indicating excessive optimism. Markets could be overheated; profit booking or hedging may be wise."

        if not api_key:
            fb = fallback_msg(label, mmi_value)
            try:
                self.store_ai_feedback(symbol, fb)
            except Exception as e:
                logging.debug(f"Failed to store fallback ai feedback: {e}")
            return fb

        # call Mistral
        try:
            prompt = (
                f"You are an unbiased financial assistant. Given only the Market Mood Index (MMI) value {mmi_value} "
                f"(range 0 to 100), give factual feedback on whether the user should buy, sell, or stay calm. "
                f"Also, state if the market mood is bullish, bearish, or neutral based purely on MMI statistics. "
                f"Keep it concise, easy to understand language, and free of bias. Do not repeat the obvious MMI number and information. "
                f"Avoid giving numbers in square brackets in your response."
            )

            url = "https://api.mistral.ai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            data = {
                "model": "mistral-medium", 
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
            # Mistral API is OpenAI-compatible, so same parsing logic
            result_text = result["choices"][0]["message"]["content"].strip() if "choices" in result and result["choices"] else str(result)

            # Clean up response
            lines = [l.strip() for l in result_text.split("\n") if l.strip()]
            if lines:
                fb = " ".join(lines[:2])
            else:
                fb = result_text

            # Additional cleanup: remove markdown, citations, etc.
            fb = fb.replace("**", "").replace("[", "").replace("]", "")
            fb = ' '.join(fb.split())

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
            cursor.execute("""DELETE FROM sector_news_cache WHERE symbol = %s AND cached_at < (NOW() - INTERVAL 1 DAY)""", (symbol,))
            max_cache_time = time.time()
            # Insert up to 10 latest
            for a in articles:
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
                "SELECT news_title, news_desc, news_url, published_at FROM sector_news_cache WHERE symbol=%s AND published_at >= FROM_UNIXTIME(%s) ORDER BY published_at DESC LIMIT 10",
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
        Recompute and store MMI for a symbol with explicit weighting, rolling windows, outlier handling, z-score normalization, and component-level contributions.
        """
        import scipy.stats
        if symbol == "NEWS":
            return None
        logging.info(f'Recomputing/storing MMI for topic: {symbol}')
        conn = self._get_connection()
        lock_cursor = conn.cursor(buffered=True)
        lock_name = f"mmi_{symbol}"
        lock_acquired = False
        now_ts = datetime.datetime.utcnow().timestamp()

        try:
            # Acquire advisory lock for this symbol (timeout 10s)
            try:
                lock_cursor.execute("SELECT GET_LOCK(%s, %s)", (lock_name, 10))
                res = lock_cursor.fetchone()
                lock_acquired = bool(res and res[0] == 1)
            except Exception as e:
                logging.warning(f"GET_LOCK failed for {symbol}: {e}")
                lock_acquired = False

            if not lock_acquired:
                logging.warning(f"Could not acquire lock for {symbol}, using cached value if available")
                existing = self.get_mmi_for_topic(symbol)
                return existing['mmi'] if existing else 0

            # After acquiring lock, re-check cached MMI freshness
            existing = self.get_mmi_for_topic(symbol)
            one_day_ago = now_ts - 86400
            if existing and existing.get('last_update') and float(existing.get('last_update')) >= one_day_ago:
                logging.info(f'Using cached MMI for {symbol} (age <24h)')
                return existing['mmi']

            # Rolling windows: 1D, 7D, 30D
            def filter_by_window(rows, days):
                cutoff = now_ts - days * 86400
                return [r for r in rows if float(r[0]) >= cutoff]

            rows_1d = filter_by_window(rows, 1)
            rows_7d = filter_by_window(rows, 7)
            rows_30d = filter_by_window(rows, 30)

            # Outlier handling: remove sentiment_score outliers (z-score > 3)
            def remove_outliers(scores):
                if len(scores) < 5:
                    return scores
                z = np.abs(scipy.stats.zscore(scores))
                return [s for s, zval in zip(scores, z) if zval < 3]

            # Source-wise weighting: news=0.5, social=0.3, other=0.2
            def weighted_sentiment(rows):
                news_scores = [float(r[5]) for r in rows if r[2]=='news' and r[5] is not None]
                social_scores = [float(r[5]) for r in rows if r[2] in ('twitter','social') and r[5] is not None]
                other_scores = [float(r[5]) for r in rows if r[2] not in ('news','twitter','social') and r[5] is not None]
                news_scores = remove_outliers(news_scores)
                social_scores = remove_outliers(social_scores)
                other_scores = remove_outliers(other_scores)
                w_news = 0.5
                w_social = 0.3
                w_other = 0.2
                comp = {
                    'news': float(np.mean(news_scores)) if news_scores else 0.0,
                    'social': float(np.mean(social_scores)) if social_scores else 0.0,
                    'other': float(np.mean(other_scores)) if other_scores else 0.0
                }
                total = w_news*comp['news'] + w_social*comp['social'] + w_other*comp['other']
                return total, comp

            # Use 1D window for current MMI
            mmi_raw, comp_1d = weighted_sentiment(rows_1d)
            # 7D and 30D for reference
            mmi_7d, comp_7d = weighted_sentiment(rows_7d)
            mmi_30d, comp_30d = weighted_sentiment(rows_30d)

            # Normalize MMI using z-score (1D window)
            all_scores = [float(r[5]) for r in rows_30d if r[5] is not None]
            if len(all_scores) > 5:
                mmi_z = float((mmi_raw - np.mean(all_scores)) / (np.std(all_scores) + 1e-6))
                mmi_norm = float(scipy.stats.norm.cdf(mmi_z))  # percentile
            else:
                mmi_norm = (mmi_raw + 1) / 2

            scaled_mmi = int(mmi_norm * 100)
            if scaled_mmi <= 25:
                mood = "Extreme Fear"
            elif scaled_mmi <= 50:
                mood = "Fearful / Cautious"
            elif scaled_mmi <= 75:
                mood = "Neutral to Optimistic"
            else:
                mood = "Extreme Greed / Euphoria"

            # ---------------- Explainability & Attribution ----------------
            def _build_contributors(rows_window, top_n=5):
                """Identify top positive/negative contributors by sentiment_score."""
                items = []
                for r in rows_window:
                    try:
                        score = float(r[5]) if r[5] is not None else 0.0
                    except Exception:
                        continue
                    text = (r[3] or "") if isinstance(r, (list, tuple)) else str(r)
                    ts = r[1] if isinstance(r, (list, tuple)) else None
                    src = r[2] if isinstance(r, (list, tuple)) else None
                    items.append({
                        "timestamp": str(ts),
                        "source": src or "",
                        "symbol": symbol,
                        "sentiment_score": score,
                        "text": (text or "")[:280],
                    })
                positives = sorted(
                    [i for i in items if i["sentiment_score"] > 0],
                    key=lambda x: -x["sentiment_score"],
                )[:top_n]
                negatives = sorted(
                    [i for i in items if i["sentiment_score"] < 0],
                    key=lambda x: x["sentiment_score"],
                )[:top_n]
                return {"top_positive": positives, "top_negative": negatives}

            contributors_1d = _build_contributors(rows_1d, top_n=5)

            # Sector-wise sentiment breakdown for this symbol (1D window)
            def _label_from_row(r):
                # sentiment_label is at index 10 if present; fall back on score
                try:
                    lbl = r[10]
                except Exception:
                    lbl = None
                if lbl:
                    return str(lbl)
                try:
                    s = float(r[5]) if r[5] is not None else 0.0
                except Exception:
                    s = 0.0
                if s > 0.05:
                    return "positive"
                if s < -0.05:
                    return "negative"
                return "neutral"

            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
            for r in rows_1d:
                label = _label_from_row(r)
                if label in sentiment_counts:
                    sentiment_counts[label] += 1

            total_labeled = max(
                1, sum(sentiment_counts.values())
            )  # avoid divide-by-zero
            sector_sentiment_breakdown = {
                "counts": sentiment_counts,
                "share": {
                    k: v / total_labeled for k, v in sentiment_counts.items()
                },
            }

            # Data freshness: how recent is the latest ingestion for this symbol
            latest_ingestion = None
            for r in rows_30d:
                try:
                    ts = float(r[9])  # ingestion_ts index
                except Exception:
                    continue
                if latest_ingestion is None or ts > latest_ingestion:
                    latest_ingestion = ts
            if latest_ingestion is not None:
                age_sec = max(0.0, now_ts - latest_ingestion)
                data_freshness = {
                    "latest_ingestion_ts": latest_ingestion,
                    "age_seconds": age_sec,
                }
            else:
                data_freshness = {
                    "latest_ingestion_ts": None,
                    "age_seconds": None,
                }

            # Source-level attribution (relative weights)
            total_abs = (
                abs(comp_1d.get("news", 0.0))
                + abs(comp_1d.get("social", 0.0))
                + abs(comp_1d.get("other", 0.0))
            )
            if total_abs <= 0:
                source_share = {"news": 0.0, "social": 0.0, "other": 0.0}
            else:
                source_share = {
                    k: abs(v) / total_abs for k, v in comp_1d.items()
                }

            # Delta vs previous MMI if available
            delta_info = None
            if existing and isinstance(existing.get("detail"), str):
                try:
                    prev_detail = json.loads(existing["detail"] or "{}")
                    prev_mmi = float(prev_detail.get("scaled_mmi", existing["mmi"]))
                    prev_comp = prev_detail.get("component_1d", {})
                    delta_info = {
                        "prev_mmi": prev_mmi,
                        "delta_mmi": scaled_mmi - prev_mmi,
                        "delta_components": {
                            src: comp_1d.get(src, 0.0)
                            - float(prev_comp.get(src, 0.0))
                            for src in ("news", "social", "other")
                        },
                    }
                except Exception as e:
                    logging.debug(
                        f"Failed to parse previous MMI detail for {symbol}: {e}"
                    )

            # Human-readable explanation summarizing main drivers
            def _summarize_drivers():
                pos = contributors_1d["top_positive"]
                neg = contributors_1d["top_negative"]
                parts = []
                if delta_info:
                    dm = delta_info.get("delta_mmi")
                    if dm is not None:
                        if dm > 0:
                            parts.append(
                                f"MMI moved up by {dm:.1f} points driven mainly by improving sentiment."
                            )
                        elif dm < 0:
                            parts.append(
                                f"MMI declined by {abs(dm):.1f} points due to rising negative sentiment."
                            )
                if pos:
                    parts.append(
                        f"Positive tone is supported by headlines such as: "
                        + "; ".join(p["text"] for p in pos[:3])
                    )
                if neg:
                    parts.append(
                        f"Negative pressure comes from headlines like: "
                        + "; ".join(n["text"] for n in neg[:3])
                    )
                if not parts:
                    parts.append(
                        "Sentiment is mixed with no single dominant driver in the latest window."
                    )
                return " ".join(parts)

            news_summary = _summarize_drivers()

            detail = json.dumps({
                'mmi_1d': mmi_raw,
                'mmi_7d': mmi_7d,
                'mmi_30d': mmi_30d,
                'component_1d': comp_1d,
                'component_7d': comp_7d,
                'component_30d': comp_30d,
                'mmi_norm': mmi_norm,
                'scaled_mmi': scaled_mmi,
                'symbol': symbol,
                'contributors_1d': contributors_1d,
                'sector_sentiment_breakdown': sector_sentiment_breakdown,
                'data_freshness': data_freshness,
                'source_share_1d': source_share,
                'delta_info': delta_info,
            })
            last_update = now_ts

            # commit MMI to DB
            mmi_cursor = conn.cursor(buffered=True)
            try:
                # Upsert latest MMI into cache
                if self.get_mmi_for_topic(symbol):
                    mmi_cursor.execute(
                        "UPDATE mmi_cache SET mmi=%s, mood=%s, detail=%s, last_update=%s, news_summary=%s WHERE symbol=%s",
                        (scaled_mmi, mood, detail, last_update, news_summary, symbol),
                    )
                else:
                    mmi_cursor.execute(
                        "INSERT INTO mmi_cache (symbol, mmi, mood, detail, last_update, news_summary) VALUES (%s, %s, %s, %s, %s, %s)",
                        (symbol, scaled_mmi, mood, detail, last_update, news_summary),
                    )
                # Also persist a daily snapshot into mmi_history (as_of = start of UTC day)
                as_of_day = (int(now_ts) // 86400) * 86400
                mmi_cursor.execute(
                    "INSERT INTO mmi_history (symbol, mmi, as_of) VALUES (%s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE mmi=%s",
                    (symbol, scaled_mmi, as_of_day, scaled_mmi),
                )
                conn.commit()
            except Exception as e:
                logging.warning(
                    f"Failed to write mmi_cache for {symbol}: {e}",
                    extra={"error_category": "database", "symbol": symbol},
                )
                try:
                    conn.rollback()
                except Exception:
                    pass
            finally:
                mmi_cursor.close()

            return scaled_mmi

        finally:
            # Always release the advisory lock if we acquired it
            try:
                if lock_acquired:
                    lock_cursor.execute("SELECT RELEASE_LOCK(%s)", (lock_name,))
                    # not checking return value, but log if needed
            except Exception as e:
                logging.error(f"Failed to release lock for {symbol}: {e}")
            try:
                lock_cursor.close()
            except Exception:
                pass

    def insert_entry(self, data):
        conn = self._get_connection()
        cursor = conn.cursor(buffered=True)
        try:
            # Check for duplicate using content_hash if present
            content_hash = data.get('content_hash')
            if content_hash:
                cursor.execute("SELECT COUNT(*) FROM sentiment_data WHERE symbol=%s AND timestamp=%s AND MD5(text) = MD5(%s)", (data['symbol'], data['timestamp'], data['text']))
                if cursor.fetchone()[0] > 0:
                    logging.info(f"Duplicate entry skipped for {data['symbol']} at {data['timestamp']}")
                    return
            # Store both raw and cleaned text if available
            cleaned_text = data.get('cleaned_text', data['text'])
            # Add columns if not exist (idempotent)
            try:
                cursor.execute("ALTER TABLE sentiment_data ADD COLUMN cleaned_text TEXT")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE sentiment_data ADD COLUMN sector VARCHAR(64)")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE sentiment_data ADD COLUMN ingestion_ts DOUBLE")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE sentiment_data ADD COLUMN sentiment_label VARCHAR(32)")
            except Exception:
                pass
            try:
                cursor.execute("ALTER TABLE sentiment_data ADD COLUMN sentiment_confidence FLOAT")
            except Exception:
                pass
            cursor.execute(
                "INSERT INTO sentiment_data (timestamp, source, text, cleaned_text, symbol, sentiment_score, sentiment_label, sentiment_confidence, emotion, sector, ingestion_ts) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (data['timestamp'], data['source'], data['text'], cleaned_text, data['symbol'], data['sentiment_score'], data.get('sentiment_label'), data.get('sentiment_confidence'), data['emotion'], data.get('sector'), data.get('ingestion_ts'))
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

    def get_mmi_history(self, symbol: str, days: int):
        """
        Fetch historical MMI snapshots for a symbol over the last `days` days.
        Returns list of (as_of, mmi) tuples ordered by as_of ascending.
        """
        cursor = self._get_cursor()
        try:
            now_ts = datetime.datetime.utcnow().timestamp()
            cutoff = now_ts - days * 86400
            cursor.execute(
                "SELECT as_of, mmi FROM mmi_history WHERE symbol=%s AND as_of >= %s ORDER BY as_of ASC",
                (symbol, cutoff),
            )
            return cursor.fetchall()
        except Exception as e:
            logging.warning(f"get_mmi_history failed for {symbol}: {e}")
            return []
        finally:
            cursor.close()