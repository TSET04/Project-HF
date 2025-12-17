import pandas as pd

from utils.mmi_backtest import compute_historical_mmi


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def fetch_recent_entries(self, limit: int = 100):
        return self._rows[:limit]


def test_compute_historical_mmi_basic_structure():
    # Build minimal fake sentiment rows for a single symbol
    rows = []
    ts_base = pd.Timestamp.now().floor("H")
    for i in range(10):
        ts = ts_base - pd.Timedelta(hours=i)
        rows.append(
            [
                ts.isoformat(),  # timestamp
                "news",  # source
                f"text {i}",  # text
                f"cleaned {i}",  # cleaned_text
                "banking",  # symbol
                0.1 * ((-1) ** i),  # sentiment_score
                "positive" if i % 2 == 0 else "negative",  # sentiment_label
                0.9,  # sentiment_confidence
                "neutral",  # emotion
                "banking",  # sector
                ts.timestamp(),  # ingestion_ts
            ]
        )

    db = _FakeDB(rows)
    df = compute_historical_mmi(db, "banking", days=2)

    assert "mmi_1d" in df.columns
    assert "mmi_7d" in df.columns
    assert "mmi_30d" in df.columns
    # We also expect a volatility proxy for confidence bands
    assert "mmi_1d_std" in df.columns
    # Should not be completely empty
    assert not df.empty


