# Market-Mood AI

> **AI-Driven Sentiment Intelligence Dashboard for Hedge Funds & Traders**

Market-Mood AI is an enterprise-grade sentiment analysis platform that processes real-time market data from financial news sources to compute a continuous Market Mood Index (MMI). Designed specifically for quantitative traders and hedge fund managers, it provides actionable sentiment metrics, trend analysis, and AI-generated market insights through an interactive Streamlit dashboard.

---

## 📋 Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [Dashboard Features](#dashboard-features)
- [Sector Configuration](#sector-configuration)
- [Technology Stack](#technology-stack)
- [API Integrations](#api-integrations)
- [Database Options](#database-options)
- [Workflow & Scheduling](#workflow--scheduling)
- [Development Guide](#development-guide)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

### Core Capabilities
- **Real-Time Sentiment Collection**: Automatically fetches financial news articles from NewsData.io API
- **FinBERT Sentiment Analysis**: Uses pre-trained FinBERT model for finance-specific sentiment scoring
- **Emotion Classification**: Tracks market sentiment indicators (Fear/Greed/Neutral)
- **Market Mood Index (MMI)**: Weighted composite sentiment score with daily updates
- **Multi-Sector Support**: Analyze specific market sectors (Banking, IT, Pharma, FMCG, Auto) with custom keyword filtering
- **Thread-Safe Data Pipeline**: Non-blocking, event-driven collection loop with background processing
- **Persistent Storage**: MySQL backend with comprehensive database schema
- **Real-Time Dashboard**: Interactive Streamlit UI with glass-morphism design
- **AI-Powered Insights**: Automatic generation of market summaries and analysis

### Dashboard Capabilities
- 📊 Live MMI gauge showing market sentiment
- 📈 Historical trend charts with sector breakdowns
- 📰 Real-time news feed and sentiment breakdown
- 🔍 Source analytics and sentiment distribution
- ⚡ Auto-refreshing components with caching
- 🎯 Sector-specific sentiment tracking

---

## 🏗️ Architecture

```
Data Collection Layer (Async)
    ├── NewsData.io API → News articles
    └── Processing Pipeline → FinBERT sentiment
        │
        ├→ Sentiment Score (-1 to +1)
        ├→ Emotion Classification
        └→ Database Storage
             │
Market Mood Index (MMI) Calculation
    ├─→ Aggregate sector sentiment
    ├─→ Normalize & weight by recency
    └─→ Cache in MySQL
         │
Interactive Dashboard (Streamlit)
    ├─→ Real-time MMI display
    ├─→ Trend visualization
    ├─→ Sector analytics
    └─→ Live news feed
```

---

## 📁 Project Structure

```
market_mood_ai/
├── main.py                          # Application entry point & global config
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── .env                            # Environment variables (NEVER commit)
│
├── models/
│   ├── sentiment.py               # FinBERT sentiment analyzer & emotion classifier
│   └── __pycache__/
│
├── utils/
│   ├── data_loader.py            # DataCollector class, collection loop, scheduling
│   ├── db.py                     # DatabaseHandler (MySQL operations, MMI caching)
│   ├── news.py                   # NewsLoader (NewsData.io API integration)
│   ├── helper.py                 # Utility functions (deduplication, retry logic)
│   └── __pycache__/
│
├── dashboard/
│   ├── app.py                    # Streamlit UI (charts, gauges, filters)
│   └── __pycache__/
│
├── market_mood/                  # Virtual environment (local)
│   ├── pyvenv.cfg
│   ├── Scripts/
│   ├── Lib/
│   └── Include/
│
└── __pycache__/
```

### Key File Responsibilities

| File | Purpose |
|------|---------|
| `main.py` | Central configuration, MMI cache initialization, service startup |
| `models/sentiment.py` | Sentiment scoring using FinBERT, emotion detection |
| `utils/data_loader.py` | Background data collection scheduler, streaming loop |
| `utils/db.py` | MySQL database abstraction, MMI calculations, caching |
| `utils/news.py` | NewsData.io API wrapper for article fetching |
| `utils/helper.py` | Text deduplication, retry sessions, text parsing |
| `dashboard/app.py` | Interactive Streamlit dashboard UI and visualizations |

---

## 🔧 Prerequisites

### System Requirements
- Python 3.8+
- MySQL 5.7+ (or compatible database)
- 2GB RAM minimum (4GB+ recommended for ML models)
- Internet connection for API calls

### Required API Keys
1. **NewsData.io API** - Financial news source
2. **MySQL Database** - Sentiment and MMI storage
3. **(Optional) Mistral AI API** - For daily market summaries

---

## 📦 Installation & Setup

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd market_mood_ai
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv market_mood
market_mood\Scripts\activate

# macOS/Linux
python3 -m venv market_mood
source market_mood/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: The first installation downloads the FinBERT model (~300MB). This is cached locally in `~/.cache/huggingface/`.

### Step 4: Database Setup
Ensure you have a MySQL database running and accessible. Create a new database:
```sql
CREATE DATABASE market_mood_ai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

---

## 🔐 Environment Configuration

### Create `.env` File
Create a `.env` file in the project root directory:

```bash
# Required: NewsData.io
NEWSDATA_API_KEY=your_newsdata_api_key_here

# Required: MySQL Database
MYSQL_HOST=localhost
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=market_mood_ai

# Optional: Mistral AI (for summaries)
MISTRAL_API_KEY=your_mistral_key_here
```

### Obtaining API Keys

**NewsData.io:**
1. Visit https://newsdata.io/
2. Sign up for a free account
3. Copy API key from dashboard
4. Set `NEWSDATA_API_KEY` in `.env`

**MySQL Setup:**
- Local: `localhost`, `root`, `password` (default)
- Cloud: Use RDS, DigitalOcean, or similar managed database
- Ensure network access is configured

**Mistral API (Optional):**
1. Visit https://www.mistral.ai/
2. Create API account and generate key
3. Set `MISTRAL_API_KEY` in `.env`

### Security Notes
```
⚠️ IMPORTANT:
- NEVER commit .env file to version control
- Add .env to .gitignore
- Use strong, unique database passwords
- Rotate API keys regularly
- Use environment variable management tools in production
```

---

## 🚀 Running the Application

### Option 1: Full Stack (Recommended)
```bash
python main.py
```

This will:
1. ✅ Initialize MySQL database and tables
2. ✅ Load MMI cache from recent entries
3. ✅ Start background data collection loop
4. ✅ Launch Streamlit dashboard on `http://localhost:8501`

### Option 2: Dashboard Only
```bash
streamlit run dashboard/app.py
```

Note: Ensure data collection is running separately or via `main.py`.

### Accessing the Dashboard
- **Local**: Open browser → `http://localhost:8501`
- **Remote**: `http://<server-ip>:8501`
- **Streamlit Cloud**: Configure environment variables and deploy

### Logs
Monitor application logs in the terminal:
```
INFO: Starting main()
INFO: DatabaseHandler initialized
INFO: MMI cache initialized
INFO: Beginning continuous data collection loop
INFO: Starting (or refreshing) dashboard UI...
```

---

## 📊 Dashboard Features

### Home Page
- **Market Mood Gauge**: Large circular indicator (0-100) showing current MMI
- **Sector Dropdown**: Select specific sector for detailed analysis
- **Key Metrics**: Latest sentiment statistics, data point counts

### Trend Analysis
- **Sentiment History Chart**: Line graph of sentiment over time
- **Sector Comparison**: Multi-sector sentiment overlay
- **Confidence Bands**: Statistical confidence intervals

### News & Sources
- **Live Feed**: Latest articles with sentiment labels
- **Source Breakdown**: Pie chart showing sentiment distribution
- **Keyword Analysis**: Word cloud of trending market terms

### Settings
- **Sector Filter**: Switch between All Sectors or individual sectors
- **Time Range**: Last 24h, 7d, 30d, all data
- **Auto-Refresh**: Toggle dashboard update interval

---

## 🎯 Sector Configuration

### Single Source of Truth
All sector configurations are centralized in `main.py` for easy maintenance:

```python
# Display names shown to users
SECTOR_DISPLAY = ['Banking', 'IT', 'Pharma', 'FMCG', 'Auto']

# Database keys for storage and retrieval
SECTOR_DB_KEYS = ['banking', 'information technology', 'pharma', 'fmcg', 'auto']

# Create mapping
SECTOR_MAP = dict(zip(SECTOR_DISPLAY, SECTOR_DB_KEYS))

# Keywords for news collection
NEWS_KEYWORDS = [
    'banking sector india',
    'it sector india',
    'pharma sector india',
    'fmcg sector india',
    'auto sector india'
]
```

### Customization Steps
1. Open `main.py` at the top of the file
2. Modify the lists above to match your sectors:
   ```python
   # Example: Add Energy sector
   SECTOR_DISPLAY = ['Banking', 'IT', 'Pharma', 'FMCG', 'Auto', 'Energy']
   SECTOR_DB_KEYS = ['banking', 'information technology', 'pharma', 'fmcg', 'auto', 'energy']
   NEWS_KEYWORDS = [..., 'energy sector india']
   ```
3. Save and restart the application
4. Changes instantly propagate to:
   - Dashboard dropdown menu
   - News collection queries
   - MMI calculation and caching
   - Database sector tags

### Adding/Removing Sectors
- **Add**: Append to all three lists (SECTOR_DISPLAY, SECTOR_DB_KEYS, NEWS_KEYWORDS)
- **Remove**: Delete from all three lists
- **Rename**: Update SECTOR_DISPLAY only; DB keys should remain consistent

---

## 🛠️ Technology Stack

### Core Libraries
| Component | Technology | Version |
|-----------|-----------|---------|
| **Web UI** | Streamlit | Latest |
| **Sentiment Analysis** | FinBERT (Transformers) | Pre-trained |
| **ML Framework** | PyTorch | With CPU/GPU support |
| **Database** | MySQL Connector | 8.x |
| **Data Processing** | Pandas + NumPy | Latest |
| **Visualization** | Plotly + Matplotlib | Latest |
| **API Requests** | Requests library | With retry logic |

### Dependencies (from requirements.txt)
```
streamlit              # Web dashboard framework
pandas, numpy          # Data manipulation
newsapi-python         # News API wrapper (legacy)
transformers, torch    # FinBERT model
scikit-learn           # ML utilities
yfinance               # Stock price data
python-dotenv          # Environment variables
matplotlib, plotly     # Visualization
mysql-connector-python # MySQL client
requests               # HTTP library
```

---

## 🔗 API Integrations

### NewsData.io
- **Endpoint**: `https://newsdata.io/api/1/news`
- **Rate Limit**: Varies by plan (typically 200 requests/day free)
- **Response**: JSON articles with title, description, link, publishDate
- **Coverage**: Global financial news, Indian markets
- **Integration**: `utils/news.py` → `NewsLoader` class

### MySQL Database
- **Type**: Relational database
- **Tables**:
  - `sentiment_data`: Raw sentiment entries
  - `mmi_cache`: Cached Market Mood Index values
  - `sector_news_cache`: News article cache
  - `indices_data`: Index price tracking
  - `trends_data`: Trending keywords
  - `ai_feedback`: Generated summaries
- **Connection**: Thread-local pooling via `utils/db.py`

---

## 💾 Database Options

### Current: MySQL (Required)
```
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=password
MYSQL_DATABASE=market_mood_ai
```

**Setup**:
```bash
# Windows
mysql -u root -p < schema.sql

# macOS/Linux
mysql -u root -p < schema.sql
```

### Alternative Options (Not Currently Implemented)
- **MongoDB**: Document-based, flexible schema
- **SQLite**: File-based, no server required
- **PostgreSQL**: Open-source relational, advanced features

**Note**: Current codebase requires MySQL. Switching databases requires modifying `utils/db.py` DatabaseHandler class.

---

## ⏰ Workflow & Scheduling

### Data Collection Loop
The background data collector runs on a fixed schedule:

```
┌─────────────────────────────────────────┐
│   Background Collection Thread          │
├─────────────────────────────────────────┤
│ ✓ Event-driven waiting (low CPU)        │
│ ✓ Non-blocking, async processing        │
│ ✓ Graceful shutdown support             │
└─────────────────────────────────────────┘
         ↓
    Every 6 Hours          Every 24 Hours
    (News Collection)      (MMI Update)
    ├─ Fetch articles      ├─ Recalculate MMI
    ├─ Run FinBERT        ├─ Update cache
    ├─ Classify emotion    ├─ Generate AI feedback
    └─ Store in DB        └─ Notify dashboard
```

### Timing Configuration
Edit `utils/data_loader.py` in the `DataCollector.__init__()` method:

```python
self.timing_config = {
    'news_collection': {
        'interval_seconds': 21600,  # 6 hours
        'last_run': 0,
    },
    'mmi_update': {
        'interval_seconds': 86400,  # 24 hours
        'last_run': 0,
    }
}
```

### Key Timing Details
- **News Collection**: 6-hour intervals (can be adjusted)
- **MMI Recalculation**: 24-hour intervals with hourly cache updates
- **Dashboard Refresh**: Streamlit auto-refresh (configurable)
- **Database Commits**: Per-operation or batch
- **Shutdown**: Graceful termination with thread cleanup

---