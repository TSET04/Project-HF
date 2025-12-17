# Market-Mood AI
Market-Mood AI is an AI-driven sentiment intelligence dashboard tailored for hedge funds and traders. It analyzes real-time sentiment from X (formerly Twitter) and financial news to compute a continuous Market Mood Index (MMI), and visualizes key metrics for smart market decision-making.

## Features
- Real-time collection of finance-related text from news (and optionally social feeds)
- Finance-specific sentiment analysis using FinBERT
- (Optional) Fear/Greed emotion tracking
- DB storage (MySQL by default): timestamp, source, text, symbol, sentiment, emotion
- Market Mood Index (weighted/normalized composite sentiment)
- Interactive Streamlit dashboard: live MMI gauge, trend charts, explainability cards, and historical MMI chart with confidence band
- Backtesting / analytics utilities for MMI vs indices

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
    - MISTRAL_API_KEY for summary (see https://docs.mistral.ai/)
    - (Optional) MongoDB URI
    - (Optional) MySQL: MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
    - If neither MongoDB nor MySQL: uses local SQLite by default

```
# Example (X/Twitter v2 Project Bearer Token)
TWITTER_BEARER_TOKEN=your_project_bearer_token
NEWSAPI_KEY=xxx
MISTRAL_API_KEY=xxx
MONGO_URI=
MYSQL_HOST=
MYSQL_USER=
MYSQL_PASSWORD=
MYSQL_DATABASE=
```

## How to Run (Local Development Runbook)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the full system (ingestion + dashboard):
   ```bash
   streamlit run main.py
   ```

   - This starts the background data collector (news ingestion, MMI recomputation)
   - And serves the dashboard at `http://localhost:8501`

3. (Optional) Run just the dashboard, reusing an already running backend:
   ```bash
   streamlit run dashboard/app.py
   ```

4. Run unit tests (Pytest):
   ```bash
   pytest -q
   ```

## Daily Summary Model
The dashboard uses [Mistral's API] for ultra-fast, high-quality daily market summaries. Sign up and get your API key from Mistral, then set `Mistral_API_KEY` in your `.env`.

## Sector Configuration: Dynamic Sectors and Keywords

All tracked sectors (dashboard dropdown, sentiment collection, MMI calculation, and news searches) are controlled from a single place: `config.py`.

**To change which sectors are analyzed:**
1. Open `config.py` and modify:
   ```python
   SECTOR_DISPLAY = ["Banking", "IT", "Pharma", "FMCG", "Auto"]
   SECTOR_DB_KEYS = ["banking", "information technology", "pharma", "fmcg", "auto"]
   ```
2. Re-run the app. All changes instantly apply to:
    - Dashboard dropdown options
    - DataCollector's news collection
    - MMI caching, bootstrap logic, and display

> No need to hunt through multiple files: change tracked sectors **in one place only** (`config.py`)!

---

## Architecture Overview

At a high level, the system is split into clear modules:

- **Ingestion (`utils/data_loader.py`, `utils/news.py`)**
  - Fetches raw finance news from external APIs on a schedule
  - Applies light validation (length, finance-related keywords, language detection)
  - Writes raw texts into the DB for downstream processing

- **Preprocessing & Inference (`models/sentiment.py`, `utils/helper.py`)**
  - Normalizes text
  - Runs FinBERT for sentiment and a lightweight rule-based `EmotionClassifier`
  - Adds `sentiment_score`, `sentiment_label`, `sentiment_confidence`, `emotion` to each record

- **Indexing & Storage (`utils/db.py`)**
  - Owns all MySQL schema creation and migrations
  - Provides insert/fetch helpers for `sentiment_data`, `mmi_cache`, `sector_news_cache`, etc.
  - Handles idempotent schema evolution and transaction boundaries with rollback on failure

- **Analytics & MMI (`utils/db.py`, `utils/mmi_backtest.py`)**
  - `DatabaseHandler.recompute_and_store_mmi` computes the production MMI, writes into `mmi_cache`
  - `utils/mmi_backtest.compute_historical_mmi` builds rolling MMI proxies for historical charts and backtests

- **Dashboard (`dashboard/app.py`)**
  - User-facing Streamlit app
  - Reads from `mmi_cache`, sentiment tables, and news cache
  - Renders live MMI gauge, AI feedback, explainability sections, and historical MMI charts with confidence bands

---

## Data Flow (End-to-End)

1. **Ingestion loop** (`DataCollector.collection_loop`)
   - On a schedule, calls `NewsLoader.fetch_news` to pull sector-specific news
   - Filters, deduplicates, and enriches each article with sentiment and emotion
   - Persists records into `sentiment_data` via `DatabaseHandler.insert_entry`

2. **MMI recomputation**
   - Periodically (`update_all_mmi`) aggregates recent sentiment for each symbol
   - `recompute_and_store_mmi`:
     - Builds 1D/7D/30D windows
     - Applies source-wise weighting, outlier removal, normalization to 0–100
     - Writes `mmi`, `mood`, `detail` JSON, `last_update`, and `news_summary` into `mmi_cache`

3. **Dashboard consumption**
   - Uses `get_mmi_for_topic`, `get_ai_feedback`, `get_sector_news`, and `fetch_recent_entries`
   - Displays both the current MMI and historically reconstructed MMI (for time range selection)
   - Surfaces top drivers, sector sentiment breakdown, and data freshness indicators.

---

## MMI Formula & Assumptions

- **Inputs**
  - Sentiment scores from FinBERT for each ingested article (roughly in \[-1, 1\])
  - Source categories: `news`, `twitter/social`, and `other`

- **Steps (simplified)**
  1. Build rolling windows: 1 day, 7 days, 30 days
  2. Remove strong outliers via z-score (> 3 standard deviations)
  3. Compute weighted mean sentiment for 1D, 7D, 30D windows:
     - Example weights: news (0.5), social (0.3), other (0.2)
  4. Normalize to a 0–100 scale using:
     - Parametric z-score + normal CDF when enough data
     - Fallback linear mapping \((score + 1) / 2\) otherwise
  5. Label the MMI into bands (e.g., Extreme Fear / Cautious / Neutral / Greed)

- **Assumptions**
  - FinBERT sentiment scores are well-calibrated and approximately symmetric
  - News and social sentiment can be linearly combined with fixed weights
  - Enough observations exist in the rolling window for z-score based normalization to behave well

---

## Known Limitations & Risks

- **Data coverage**: News APIs can miss important local or niche events; social data may be incomplete.
- **Model bias**: FinBERT is trained on English financial text and may misinterpret slang or evolving jargon.
- **Latency & drift**: Sentiment is computed as data arrives; market reactions can be faster than the ingestion window.
- **Normalization assumptions**: Z-score normalization assumes a roughly stable distribution; in stressed markets this can break down.
- **No trading advice**: MMI is an informational indicator only and should not be treated as direct buy/sell advice.

---

# Safety: _Keep your API keys safe! Never commit your .env file._

---

For issues or contributions, open an issue or PR on Github.