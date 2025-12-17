import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import time
import json
from typing import Dict

from utils.db import DatabaseHandler
from config import SECTOR_MAP


import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Silences SettingWithCopy
import logging

def run_dashboard(db: DatabaseHandler, sector_map: Dict[str, str] = SECTOR_MAP) -> None:
    logging.info('Starting (or refreshing) dashboard UI...')
    st.set_page_config(page_title="Market Mood AI", layout="wide")
    if db is None:
        db = DatabaseHandler()
        logging.info('Dashboard created new DatabaseHandler')
    st.markdown(
        '''
        <style>
        body, .stApp {
            background: linear-gradient(120deg, #161b22 0%, #191d26 100%) !important; 
            color: #dde2f5;
        }
        .glass-card {
            background: rgba(36,38,58, 0.62);
            box-shadow: 0 8px 36px 0 #1e233844, 0 1.5px 24px -10px #63cae811;
            border-radius: 18px;
            padding: 32px 38px 30px 38px;
            margin-bottom: 32px;
            backdrop-filter: blur(12px);
        }
        .section-div {
            margin-top:30px; margin-bottom:22px; width:100%; border-radius:16px;
            background: rgba(44, 49, 80, 0.5); box-shadow: 0 1px 18px #201e364a;
            padding: 38px 36px 15px 36px;
        }
        .stPlotlyChart, .st-bar-chart {
            background: rgba(22, 25, 34, 0.8) !important;
            border-radius: 22px !important;
            box-shadow: 0 3px 32px -10px #222b45;
        }
        .stSelectbox label {
            color:#f7d774; font-size:1.19em; font-weight:700; margin-bottom: 0.5em;
            letter-spacing: 0.04em;
        }
        .stSelectbox {
            background: #21273c !important; border-radius:13px !important;
            border: 1.5px solid #53c9f9 !important;
            box-shadow: 0 0 14px #32e8ef11 inset;
            padding: 4px !important;
        }
        .section-title {
            color: #72e4ff; font-size:1.35em; font-weight:900; margin-bottom:0.2em;
            letter-spacing: 0.03em; font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
        }
        .subtitle {
            color: #7aa7e9; font-size:1.18em;
        }
        .stTitle, .stMarkdown h1 {
            color: #ccdfff; font-weight: 800; margin-bottom: 0.35em; text-shadow:0 1.5px 28px #61e9ed09;
            font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
        }
        .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
            color: #a6e3ff; font-family: 'Segoe UI', 'Roboto', Arial, sans-serif; font-weight: 700;
        }
        .stProgress > div > div > div > div { background-color: #09c6ee !important; height: 18px; border-radius:7px !important; }
        .stInfo { border-left: 4px solid #53c9f9; background: rgba(29,34,53,0.98) !important; color: #94b3ff; border-radius:12px; font-size: 1.09em; padding:14px 22px 14px 20px; margin-bottom: 20px;}
        .st-bb { border: none; }
        .stMarkdown h3, .stMarkdown h4, .stMarkdown h2, .stMarkdown h1 {
            font-family: 'Segoe UI', 'Roboto', Arial, sans-serif; font-weight:900; letter-spacing: 0.015em;
        }
        .stSelectbox select:focus, .stSelectbox select:active,
        .stSelectbox input:focus, .stSelectbox input:active,
        .stSelectbox [data-baseweb='select'] *:focus, .stSelectbox [data-baseweb='select'] *:active {
            outline: none !important;
            box-shadow: none !important;
            border-color: #7ddffa !important;
        }
        .stSelectbox:focus-within { box-shadow: 0 0 0 2px #7ddffa80 !important; border-color: #7ddffa !important; }
        </style>
        ''', unsafe_allow_html=True)
    st.markdown("""
        <div style='padding-bottom: 7px;'></div>
    """, unsafe_allow_html=True)
    st.title("📊 Market-Mood AI Dashboard")
    st.markdown("<div class='subtitle'>Live sector-wise Indian market mood analysis</div>", unsafe_allow_html=True)

    # Sector configuration
    sector_db_keys = list(sector_map.keys())

    # BOOTSTRAP: Ensure MMIs are computed for all sectors before rendering UI (only if stale >24h)
    if 'mmi_bootstrap_done' not in st.session_state or not st.session_state['mmi_bootstrap_done']:
        with st.spinner('Calculating MMIs for all sectors...'):
            try:
                all_rows_boot = db.fetch_recent_entries(1500)
                now_ts = time.time()
                for sym in sector_db_keys:
                    rec = db.get_mmi_for_topic(sym)
                    if not rec or (now_ts - rec['last_update']) >= 86400:
                        rows_for_sym = [row for row in all_rows_boot if (row[4] if isinstance(row, (list, tuple)) else row.get('symbol')) == sym]
                        mmival = db.recompute_and_store_mmi(sym, rows_for_sym)
                        try:
                            db.generate_and_store_ai_feedback(sym, mmival)
                        except Exception as e:
                            logging.warning(f'Failed to generate/store AI feedback for {sym} during dashboard bootstrap: {e}')
                    else:
                        logging.info(f'Skip bootstrap recompute for {sym}; cached MMI fresh (<24h).')
                        # ensure ai feedback exists for this symbol
                        try:
                            ai = db.get_ai_feedback(sym)
                            if not ai or (not ai.get('cached_at')) or float(ai.get('cached_at')) < (now_ts - 86400):
                                mmival = int(rec['mmi']) if rec and rec.get('mmi') is not None else 50
                                try:
                                    db.generate_and_store_ai_feedback(sym, mmival)
                                except Exception as e:
                                    logging.warning(f'Failed to generate/store AI feedback for {sym} during dashboard bootstrap: {e}')
                        except Exception as e:
                            logging.warning(f'Error ensuring AI feedback for {sym} during dashboard bootstrap: {e}')
                logging.info('Initial MMI bootstrap complete (only outdated recomputed).')
            except Exception as e:
                logging.error(f'Error during initial MMI bootstrap: {e}')
            st.session_state['mmi_bootstrap_done'] = True
        # Use st.rerun() if available, else fallback to return
        if hasattr(st, 'rerun'):
            st.rerun()
        else:
            return

    selected_sector = st.selectbox('Select Sector', sector_db_keys, key='sector_select')
    if not selected_sector:
        st.info("Please select a sector above to view the dashboard.")
        st.stop()
    symbol_key = sector_map[selected_sector]

    all_rows = db.fetch_recent_entries(500)

    def rows_to_df(rows):
        """Normalize raw DB rows into a dataframe with stable columns used by the UI."""
        cols = ['id', 'timestamp', 'source', 'text', 'symbol', 'sentiment_score', 'emotion']
        if not rows:
            logging.warning('Dashboard received empty data rows!')
            return pd.DataFrame([], columns=cols)

        # Rows coming directly from MySQL are tuples with many columns;
        # select only the indices we care about, guarding against schema changes.
        if isinstance(rows[0], (list, tuple)):
            raw = pd.DataFrame(rows)
            out = pd.DataFrame()
            index_map = [
                ('id', 0),
                ('timestamp', 1),
                ('source', 2),
                ('text', 3),
                ('symbol', 4),
                ('sentiment_score', 5),
                ('emotion', 6),
            ]
            for name, idx in index_map:
                out[name] = raw.iloc[:, idx] if idx < raw.shape[1] else None
            return out

        # Dict-like rows (if ever used) – ensure required columns exist
        df = pd.DataFrame(rows)
        for c in cols:
            if c not in df.columns:
                df[c] = None
        return df[cols]

    # --- Build dataframes and sector slice ---
    df = rows_to_df(all_rows)
    logging.info(f'Dashboard sector selected: {selected_sector} (DB key: {symbol_key})')

    # Fetch MMI and AI feedback from DB
    mmi_rec = db.get_mmi_for_topic(symbol_key)
    mmi_info = {
        'mmi': float(mmi_rec['mmi']) if mmi_rec and mmi_rec.get('mmi') is not None else 50.0,
        'mood': mmi_rec['mood'] if mmi_rec and mmi_rec.get('mood') else 'Neutral',
        'detail': mmi_rec['detail'] if mmi_rec and mmi_rec.get('detail') else '{}',
        'news_summary': mmi_rec['news_summary'] if mmi_rec and mmi_rec.get('news_summary') else ''
    }

    # MMI Gauge + AI feedback
    with st.container():
        st.markdown("""<div class='glass-card'>""", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>Market Mood Index (MMI) for <b>{selected_sector}</b>: <i>{mmi_info['mood']}</i></div>", unsafe_allow_html=True)
        st.progress(mmi_info['mmi']/100.0)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = mmi_info['mmi'],
            title = {'text': f"Market Mood Index (MMI)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'steps' : [
                    {'range': [0, 30], 'color': '#ff6670'},
                    {'range': [30, 70], 'color': '#ffd966'},
                    {'range': [70, 100], 'color': '#9afd8d'}
                ],
                'bar': {'color':'#5370ff'}
            }
        ))
        st.plotly_chart(fig_gauge, width='stretch')

        # AI feedback read-only from DB
        try:
            ai_cache = db.get_ai_feedback(symbol_key)
            if ai_cache and ai_cache.get('cached_at') and float(ai_cache.get('cached_at')) >= (time.time() - 86400):
                ai_feedback = ai_cache.get('feedback')
            else:
                ai_feedback = 'AI feedback not available yet. It will be generated after MMI bootstrap.'
        except Exception as e:
            logging.warning(f'Error reading AI feedback: {e}')
            ai_feedback = 'AI feedback unavailable.'

        st.markdown(f"""
            <div style='margin-top:12px; padding:12px; border-radius:10px; background: rgba(11,16,28,0.6);'>
            <strong style='color:#9ef0b0;'>AI feedback (MMI-based):</strong>
            <div style='margin-top:6px; color:#d7eaff;'>{ai_feedback}</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""</div>""", unsafe_allow_html=True)

    # MMI definition & explainability card
    with st.container():
        st.markdown("""<div class='section-div'>""", unsafe_allow_html=True)
        st.markdown("#### What is the Market Mood Index (MMI)?")
        st.write("""The Market Mood Index (MMI) is a simple number that indicates whether online sentiment and market indicators are positive or negative for a sector. It combines news sentiment, Google Trends-based user sentiment, and index performance into a single normalized score."""
        )
        # Parse stored detail JSON (if available) and display underlying parameters
        try:
            detail_json = json.loads(mmi_info.get('detail') or '{}')
            # Extract component scores, attribution, and index map
            news_s = detail_json.get('news_sentiment', None)
            trend_s = detail_json.get('trend_sentiment', None)
            idx_perf = detail_json.get('index_perf_score', None)
            idx_map = detail_json.get('index_map', {})

            # Render parameters as three small containers
            st.markdown('<div style="display:flex; gap:12px; margin-top:8px;">', unsafe_allow_html=True)        

            # Underlying indices list as small light cards
            if idx_map:
                friendly = {
                    '^NSEI': 'NIFTY 50',
                    '^BSESN': 'SENSEX (BSE)',
                    '^NSEBANK': 'NIFTY Bank',
                    '^CNXIT': 'NIFTY IT',
                    '^CNXFMCG': 'NIFTY FMCG'
                }
                st.markdown('**Underlying indices used for the index component (last fetched % change):**')
                # fixed order of indices -> single row with 5 columns
                order = ['^NSEI', '^BSESN', '^NSEBANK', '^CNXIT', '^CNXFMCG']
                cols = st.columns(5)
                for idx, ticker in enumerate(order):
                    col = cols[idx]
                    v = idx_map.get(ticker)
                    name = friendly.get(ticker, ticker)
                    if v is None:
                        col.markdown(
                            f"<div style='background:#f6f9fc; color:#072b4f; padding:10px; border-radius:10px; text-align:center;'>"
                            f"<div style='font-weight:700'>{name}</div>"
                            f"<div style='margin-top:6px; font-size:1.05em'>N/A</div>"
                            f"</div>", unsafe_allow_html=True)
                    else:
                        pct = v * 100.0
                        col.markdown(
                            f"<div style='background:#f6f9fc; color:#072b4f; padding:10px; border-radius:10px; text-align:center;'>"
                            f"<div style='font-weight:700'>{name}</div>"
                            f"<div style='margin-top:6px; font-size:1.05em'>{pct:.2f}%</div>"
                            f"</div>", unsafe_allow_html=True)

            # Additional explainability: sector sentiment breakdown & data freshness
            sector_breakdown = detail_json.get('sector_sentiment_breakdown') or {}
            freshness = detail_json.get('data_freshness') or {}
            contrib = (detail_json.get('contributors_1d') or {})

            # Sector-wise sentiment breakdown (counts)
            counts = sector_breakdown.get('counts') or {}
            if counts:
                total = max(1, sum(counts.values()))
                pos = counts.get('positive', 0)
                neg = counts.get('negative', 0)
                neu = counts.get('neutral', 0)
                st.markdown("**Sector-wise sentiment breakdown (last 24h):**")
                cols_sb = st.columns(3)
                cols_sb[0].metric("Positive", f"{pos}", f"{(pos/total)*100:.0f}%")
                cols_sb[1].metric("Neutral", f"{neu}", f"{(neu/total)*100:.0f}%")
                cols_sb[2].metric("Negative", f"{neg}", f"{(neg/total)*100:.0f}%")

            # Data freshness indicator
            latest_ts = freshness.get('latest_ingestion_ts')
            if latest_ts:
                age_sec = float(freshness.get('age_seconds') or 0)
                age_min = age_sec / 60.0
                st.markdown(
                    f"<div style='margin-top:10px; color:#9ef0b0;'>"
                    f"Last data ingestion for this sector was <b>{age_min:.1f} minutes</b> ago."
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Top contributing headlines/posts (explainability)
            top_pos = contrib.get('top_positive') or []
            top_neg = contrib.get('top_negative') or []
            if top_pos or top_neg or mmi_info.get('news_summary'):
                st.markdown("---")
                st.markdown("#### Why did the MMI move?")
                if mmi_info.get('news_summary'):
                    st.markdown(
                        f"<div style='color:#d7eaff;'>{mmi_info['news_summary']}</div>",
                        unsafe_allow_html=True,
                    )
                if top_pos:
                    st.markdown("**Top positive drivers (recent):**")
                    for item in top_pos[:3]:
                        st.markdown(f"- {item.get('text','')}")
                if top_neg:
                    st.markdown("**Top negative drivers (recent):**")
                    for item in top_neg[:3]:
                        st.markdown(f"- {item.get('text','')}")
        except Exception as e:
            logging.debug(f'Could not parse MMI detail JSON: {e}')
        st.markdown("""</div>""", unsafe_allow_html=True)

    # Top 3 news for sector
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### Top News ")
    sector_news = db.get_sector_news(symbol_key)
    if sector_news and len(sector_news) > 0:
        for news in sector_news[:3]:
            title, desc, url, published = news
            st.markdown(f"**[{title}]({url})**  ", unsafe_allow_html=True)
            if desc:
                st.markdown(f"<span style='color:#b5cafd;'>{desc}</span>", unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.info('No recent news found for this sector.')
    st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------- MMI over time (analytics) ----------------------- #
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### Historical MMI & Confidence Band")

    # Fetch up to 90 days of history from DB (daily snapshots)
    history_rows = db.get_mmi_history(symbol_key, days=90)
    if not history_rows:
        st.info("Not enough historical MMI data stored yet to show trends.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    hist_df = pd.DataFrame(history_rows, columns=["as_of", "mmi"])
    hist_df["timestamp"] = pd.to_datetime(hist_df["as_of"], unit="s")
    hist_df = hist_df.sort_values("timestamp")

    if hist_df.empty:
        st.info("Not enough historical MMI data stored yet to show trends.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Determine available windows (7/30/90) based on actual span of history
    total_span_days = (hist_df["timestamp"].iloc[-1] - hist_df["timestamp"].iloc[0]).days or 0
    candidate_windows = [7, 30, 90]
    available_windows = [d for d in candidate_windows if total_span_days >= d]

    if not available_windows:
        st.info(
            f"Historical MMI is stored, but less than 7 days are available so far "
            f"(~{total_span_days} days). Chart will appear once more history accumulates."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    col_left, col_right = st.columns(2)
    with col_left:
        range_days = st.selectbox(
            "Time range",
            options=available_windows,
            format_func=lambda x: f"Last {x} days",
            index=min(1, len(available_windows) - 1),  # prefer middle option when possible
            key="mmi_range_days",
        )
    with col_right:
        # Kept for future extension; currently source-wise history is aggregated in one MMI series.
        st.write("")
        st.caption(f"Showing stored MMI snapshots for the last {range_days} days.")

    # Filter history to the selected window
    cutoff = hist_df["timestamp"].iloc[-1] - pd.Timedelta(days=range_days)
    hist_plot = hist_df[hist_df["timestamp"] >= cutoff].copy()

    if hist_plot.empty:
        st.info("No historical MMI data available for the selected time range.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Build a simple confidence band using a rolling standard deviation over the stored MMI values
    hist_plot = hist_plot.sort_values("timestamp")
    mmi_series = hist_plot["mmi"].astype(float)
    mmi_rolling = mmi_series.rolling(window=5, min_periods=2)
    mmi_mean = mmi_rolling.mean()
    mmi_std = mmi_rolling.std()

    fig_mmi = go.Figure()
    fig_mmi.add_trace(
        go.Scatter(
            x=hist_plot["timestamp"],
            y=mmi_series,
            mode="lines",
            name="MMI (daily snapshot)",
            line=dict(color="#74cae9"),
        )
    )

    if mmi_std.notna().any():
        upper = (mmi_mean + 1.96 * mmi_std).clip(0, 100)
        lower = (mmi_mean - 1.96 * mmi_std).clip(0, 100)
        fig_mmi.add_trace(
            go.Scatter(
                x=hist_plot["timestamp"],
                y=upper,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_mmi.add_trace(
            go.Scatter(
                x=hist_plot["timestamp"],
                y=lower,
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                name="Confidence band",
                hoverinfo="skip",
                fillcolor="rgba(116, 202, 233, 0.2)",
            )
        )

    fig_mmi.update_layout(
        margin=dict(l=20, r=20, t=25, b=18),
        plot_bgcolor="#181d29",
        paper_bgcolor="#181d29",
        font_color="#d3e0ff",
        font_size=16,
        yaxis=dict(range=[0, 100], title="Scaled MMI (0–100)"),
    )
    st.plotly_chart(fig_mmi, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
