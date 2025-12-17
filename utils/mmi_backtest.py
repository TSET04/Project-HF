import pandas as pd
from scipy.stats import pearsonr
from typing import Protocol

from config import HISTORICAL_MMI_DAYS


class _DBLike(Protocol):
    def fetch_recent_entries(self, limit: int = 100) -> list:
        ...


def compute_historical_mmi(db: _DBLike, symbol: str, days: int = HISTORICAL_MMI_DAYS) -> pd.DataFrame:
    """
    Compute a historical, rolling MMI proxy based on sentiment scores.

    This is used for analytics / dashboard charting, not for the
    production mmi_cache writes (which use DatabaseHandler.recompute_and_store_mmi).
    """
    rows = db.fetch_recent_entries(10000)
    rows = [r for r in rows if (isinstance(r, (list, tuple)) and r[4] == symbol)]
    if not rows:
        return pd.DataFrame(
            columns=[
                'timestamp',
                'source',
                'text',
                'cleaned_text',
                'symbol',
                'sentiment_score',
                'sentiment_label',
                'sentiment_confidence',
                'emotion',
                'sector',
                'ingestion_ts',
                'mmi_1d',
                'mmi_7d',
                'mmi_30d',
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            'timestamp',
            'source',
            'text',
            'cleaned_text',
            'symbol',
            'sentiment_score',
            'sentiment_label',
            'sentiment_confidence',
            'emotion',
            'sector',
            'ingestion_ts',
        ],
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp')
    df = df[df['timestamp'] > pd.Timestamp.now() - pd.Timedelta(days=days)]

    if df.empty:
        return df

    # Rolling MMI (mean sentiment) and a simple volatility proxy for confidence bands
    df['mmi_1d'] = df['sentiment_score'].rolling('1D', on='timestamp').mean()
    df['mmi_1d_std'] = df['sentiment_score'].rolling('1D', on='timestamp').std()
    df['mmi_7d'] = df['sentiment_score'].rolling('7D', on='timestamp').mean()
    df['mmi_30d'] = df['sentiment_score'].rolling('30D', on='timestamp').mean()
    return df

def backtest_mmi_vs_index(mmi_df, index_df):
    # Assume index_df has columns ['date','close']
    merged = pd.merge(mmi_df, index_df, left_on='timestamp', right_on='date', how='inner')
    merged['returns'] = merged['close'].pct_change()
    corr, pval = pearsonr(merged['mmi_1d'].dropna(), merged['returns'].dropna())
    print(f'Correlation (MMI vs returns): {corr:.2f}, p={pval:.3g}')
    return {'correlation': corr, 'p_value': pval}

def lead_lag_analysis(mmi_df, index_df, max_lag=5):
    # Compute correlation for lags
    results = {}
    for lag in range(-max_lag, max_lag+1):
        shifted = mmi_df['mmi_1d'].shift(lag)
        corr, pval = pearsonr(shifted.dropna(), index_df['close'].pct_change().dropna())
        results[lag] = {'correlation': corr, 'p_value': pval}
    print('Lead-lag analysis:', results)
    return results

def save_backtest_results(results, path='backtest_results.json'):
    import json
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
