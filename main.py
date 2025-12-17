import os
from dotenv import load_dotenv
import time
import logging
from typing import Iterable

from utils.helper import setup_logging
from utils.db import DatabaseHandler
from dashboard.app import run_dashboard
from utils.data_loader import start_streams
from config import SECTOR_MAP, INDICES

def initialize_mmi_cache(db: DatabaseHandler, limit: int = 1000) -> None:
    """
    Warm up the MMI cache and AI feedback for all symbols seen
    in the most recent sentiment entries.

    This is idempotent and safe to call at startup.
    """
    logging.info("Initializing MMI cache from recent DB entries...")
    all_rows = db.fetch_recent_entries(limit)
    symbols = {row[4] if isinstance(row, (list, tuple)) else row["symbol"] for row in all_rows}
    for sym in symbols:
        rec = db.get_mmi_for_topic(sym)
        df_topic = [row for row in all_rows if (row[4] if isinstance(row, (list, tuple)) else row['symbol'])==sym]
        if not rec or (time.time() - rec['last_update']) >= 86400:
            mmival = db.recompute_and_store_mmi(sym, df_topic)
            logging.info(f"MMI initialized for {sym}: {mmival}")
        else:
            mmival = int(rec['mmi'])
            logging.info(f'Using existing MMI for {sym}: {mmival}')

        # Ensure AI feedback exists and is fresh (<24h). Generate/store if missing/stale.
        try:
            ai = db.get_ai_feedback(sym)
            now_ts = time.time()
            if not ai or (not ai.get('cached_at')) or float(ai.get('cached_at')) < (now_ts - 86400):
                try:
                    db.generate_and_store_ai_feedback(sym, mmival)
                except Exception as e:
                    logging.warning(f'Failed to generate/store AI feedback for {sym}: {e}')
        except Exception as e:
            logging.warning(f'Error while ensuring AI feedback for {sym}: {e}')

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

def main(indices: Iterable[tuple[str, str]] = INDICES) -> None:
    """
    Entry point for the Market-Mood AI system.

    - Sets up structured JSON logging
    - Initializes DB schema and MMI cache
    - Starts background ingestion / MMI update loop
    - Runs the Streamlit dashboard
    """
    # Initialize structured logging once at process start
    setup_logging()
    logging.info("Starting main()", extra={"pipeline_step": "startup"})
    db = DatabaseHandler(list(indices))
    logging.info('DatabaseHandler initialized', extra={"pipeline_step": "startup"})
    initialize_mmi_cache(db)
    logging.info('MMI cache initialized', extra={"pipeline_step": "startup"})
    # Start background collector (idempotent)
    start_streams(sector_map=SECTOR_MAP, indices=list(indices))
    logging.info(
        'Data stream collection requested (start_streams)',
        extra={"pipeline_step": "startup"}
    )
    run_dashboard(db=db, sector_map=SECTOR_MAP)

if __name__ == "__main__":
    main()
