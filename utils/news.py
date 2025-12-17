import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()


class NewsLoader:
    def __init__(self, keywords):
        self.api_key = os.environ.get('NEWSDATA_API_KEY')
        self.keywords = [keyword + " sector India" for keyword in keywords]

    def fetch_news(self, count=15):
        from utils.helper import helper
        articles = []
        session = helper.get_retry_session(retries=5, backoff_factor=2)
        for keyword in self.keywords:
            try:
                query = keyword.replace(" ", "+")
                url = (
                    f"https://newsdata.io/api/1/news?"
                    f"apikey={self.api_key}"
                    f"&q={query}"
                    f"&language=en"
                    f"&country=in"
                )
                resp = session.get(url, timeout=20)
                resp.raise_for_status()
                data = resp.json()

                symbol = keyword.split(" ")[0]
                for item in data.get('results', [])[:count]:
                    articles.append({
                        'timestamp': item.get('pubDate'),
                        'text': (item.get('title', '') or '') + ' ' + (item.get('description', '') or ''),
                        'symbol': symbol
                    })
                logging.info(
                    f"Fetched {len(data.get('results', []))} news for '{keyword}'",
                    extra={"pipeline_step": "news_api", "symbol": symbol},
                )
            except Exception as e:
                logging.warning(
                    f"NewsData.io error for '{keyword}': {e}",
                    extra={"error_category": "api"},
                )
        return articles

    def fetch_raw_news(self, q, count=25):
        from utils.helper import helper
        articles = []
        session = helper.get_retry_session(retries=5, backoff_factor=2)
        try:
            query = q.replace(" ", "+")
            url = (
                f"https://newsdata.io/api/1/news?"
                f"apikey={self.api_key}"
                f"&q={query}"
                f"&language=en"
                f"&country=in"
            )
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get('results', [])[:count]:
                articles.append({
                    'title': item.get('title', ''),
                    'description': item.get('description', ''),
                    'url': item.get('link', ''),
                    'publishedAt': item.get('pubDate', '')
                })
            logging.info(
                f"Fetched {len(data.get('results', []))} raw news for '{q}'",
                extra={"pipeline_step": "news_api"},
            )
        except Exception as e:
            logging.warning(
                f"NewsData.io error for '{q}': {e}",
                extra={"error_category": "api"},
            )
        return articles
