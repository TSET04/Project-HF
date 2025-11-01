import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class NewsLoader:
    def __init__(self, keywords):
        self.api_key = os.environ.get('NEWSDATA_API_KEY')
        self.keywords = keywords

    def fetch_news(self, count=20):
        articles = []
        for keyword in self.keywords:
            try:
                query = keyword.replace(" ", "+")  # simpler query for NewsData
                url = (
                    f"https://newsdata.io/api/1/news?"
                    f"apikey={self.api_key}"
                    f"&q={query}"
                    f"&language=en"
                    f"&country=in"
                )
                resp = requests.get(url)
                resp.raise_for_status()
                data = resp.json()

                for item in data.get('results', [])[:count]:
                    articles.append({
                        'timestamp': item.get('pubDate'),
                        'text': (item.get('title', '') or '') + ' ' + (item.get('description', '') or ''),
                        'symbol': 'NEWS'
                    })
            except Exception as e:
                print(f"NewsData.io error for '{keyword}': {e}")
        return articles

    def fetch_raw_news(self, q, count=10):
        articles = []
        try:
            query = q.replace(" ", "+")
            url = (
                f"https://newsdata.io/api/1/news?"
                f"apikey={self.api_key}"
                f"&q={query}"
                f"&language=en"
                f"&country=in"
            )
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get('results', [])[:count]:
                articles.append({
                    'title': item.get('title', ''),
                    'description': item.get('description', ''),
                    'url': item.get('link', ''),
                    'publishedAt': item.get('pubDate', '')
                })
        except Exception as e:
            print(f"NewsData.io error for '{q}': {e}")
        return articles
