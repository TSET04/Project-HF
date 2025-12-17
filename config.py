"""
Global configuration for Market-Mood AI.

This module centralizes tunable constants so that hard‑coded values
are not scattered across the codebase.
"""

from __future__ import annotations

from typing import Final, List, Tuple

# -------------------------- Sectors & Indices --------------------------- #

# Display names in the dashboard
SECTOR_DISPLAY: Final[List[str]] = ["Banking", "IT", "Pharma", "FMCG", "Auto"]

# Corresponding DB / symbol keys
SECTOR_DB_KEYS: Final[List[str]] = [
    "banking",
    "information technology",
    "pharma",
    "fmcg",
    "auto",
]

# Mapping display name -> DB key
SECTOR_MAP: Final[dict[str, str]] = dict(zip(SECTOR_DISPLAY, SECTOR_DB_KEYS))

# Underlying indices tracked along with MMI
INDICES: Final[List[Tuple[str, str]]] = [
    ("^NSEI", "NIFTY 50"),
    ("^BSESN", "BSE SENSEX"),
    ("^NSEBANK", "NIFTY BANK"),
    ("^CNXIT", "NIFTY IT"),
    ("^CNXFMCG", "NIFTY FMCG"),
]

# --------------------- Collection & Ingestion Tuning -------------------- #

# News collection interval (seconds)
NEWS_COLLECTION_INTERVAL_SECONDS: Final[int] = 6 * 60 * 60  # 6 hours

# MMI recomputation interval (seconds)
MMI_UPDATE_INTERVAL_SECONDS: Final[int] = 24 * 60 * 60  # 24 hours

# Minimum length for an article to be considered (characters)
MIN_ARTICLE_LENGTH: Final[int] = 30

# Minimum sentiment confidence threshold to keep a prediction
MIN_SENTIMENT_CONFIDENCE: Final[float] = 0.6

# Default fetch window for news (minutes); None means unbounded
DEFAULT_NEWS_FETCH_WINDOW_MINUTES: Final[int | None] = None

# Rolling-window history horizon for analytical MMI (days)
HISTORICAL_MMI_DAYS: Final[int] = 90

# ----------------------------- MMI Config ------------------------------- #

# Thresholds on scaled MMI (0..100)
MMI_EXTREME_FEAR_MAX: Final[int] = 25
MMI_FEARFUL_MAX: Final[int] = 50
MMI_NEUTRAL_OPTIMISTIC_MAX: Final[int] = 75


