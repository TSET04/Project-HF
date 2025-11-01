# Market-Mood AI

Market-Mood AI is an AI-driven sentiment intelligence dashboard tailored for hedge funds and traders. It analyzes real-time sentiment from X (formerly Twitter) and financial news to compute a continuous Market Mood Index (MMI), and visualizes key metrics for smart market decision-making.

## Features
- Real-time collection of finance-related text from X (Twitter API v2/Bearer), and news (NewsAPI)
- Finance-specific sentiment analysis using FinBERT
- (Optional) Fear/Greed emotion tracking
- DB storage (SQLite/MongoDB/MySQL): timestamp, source, text, symbol, sentiment, emotion
- Market Mood Index (weighted/normalized composite sentiment)
- Interactive Streamlit dashboard: live MMI gauge, trend charts, source breakdown, live feed, auto daily summary
- Optional: Overlay MMI vs. stock indices with yfinance

## Directory Structure
```
market_mood_ai/
├── data/
├── models/
├── utils/
├── dashboard/
├── main.py
├── requirements.txt
└── README.md
```

---

## Getting Started
See below sections for setup, environment variables, and usage.

## Setup (Environment Variables)

1. Create a file named `.env` in the `market_mood_ai/` directory (see `.env.example`) and fill in:
    - X (Twitter) API Bearer Token (`TWITTER_BEARER_TOKEN` from your Developer Portal Project)
    - NewsAPI key
    - Perplexity API key for summary (see https://docs.perplexity.ai/)
    - (Optional) MongoDB URI
    - (Optional) MySQL: MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
    - If neither MongoDB nor MySQL: uses local SQLite by default

```
# Example (X/Twitter v2 Project Bearer Token)
TWITTER_BEARER_TOKEN=your_project_bearer_token
NEWSAPI_KEY=xxx
PERPLEXITY_API_KEY=xxx
MONGO_URI=
MYSQL_HOST=
MYSQL_USER=
MYSQL_PASSWORD=
MYSQL_DATABASE=
```

## How to Run

```
pip install -r requirements.txt
# Start the main pipeline (data collection runs in background)
python main.py
```

- The dashboard auto-launches on `localhost:8501` (or as streamlit app if run via `streamlit run dashboard/app.py`).
- For Streamlit Cloud, push this repo and set env variables in deployment settings.

## Daily Summary Model
The dashboard uses [Perplexity's API](https://docs.perplexity.ai/) for ultra-fast, high-quality daily market summaries. Sign up and get your API key from Perplexity, then set `PERPLEXITY_API_KEY` in your `.env`.

## Sector Configuration: Dynamic Sectors and Keywords

All tracked sectors (dashboard dropdown, sentiment collection, MMI calculation, and news searches) are controlled from a single place: `main.py`.

**To change which sectors are analyzed:**
1. Open `main.py` and modify these variables at the top:
```
SECTOR_DISPLAY = ['Banking','IT','Pharma','FMCG','Auto']   # Names as displayed to users
SECTOR_DB_KEYS = ['banking','information technology','pharma','fmcg','auto']   # Database/search keys
SECTOR_MAP = dict(zip(SECTOR_DISPLAY, SECTOR_DB_KEYS))     # Mapping of display name to DB key
NEWS_KEYWORDS = ['banking sector india','it sector india','pharma sector india','fmcg sector india','auto sector india']
```
2. Update these lists/dictionaries to fit your custom sector breakdown. For example, add "Energy" or rename "IT" to "Tech".
3. Re-run the app. All changes instantly apply to:
    - Dashboard dropdown options
    - DataCollector's X/Twitter and News collection
    - MMI caching, bootstrap logic, and display

> No need to hunt through multiple files: change tracked sectors **in one place only** (main.py)!

---

# Safety: _Keep your API keys safe! Never commit your .env file._

---

For issues or contributions, open an issue or PR on Github.