"""AI-Powered Pairs Trading & Fundamental Gap Analyzer
Main Streamlit Application.
"""

import hashlib
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pandas as pd
import streamlit as st

from src.config import (
    PEER_PRESETS,
    OPENROUTER_FREE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_LOOKBACK_PERIOD,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_ENTRY_Z,
    DEFAULT_EXIT_Z,
    DEFAULT_STOP_LOSS_Z,
    DEFAULT_TRANSACTION_COST_BPS,
    DEFAULT_DL_EPOCHS,
    DEFAULT_DL_BATCH_SIZE,
    DEFAULT_DL_LR,
    DEFAULT_DL_PRED_HORIZON,
)
from src.data import (
    fetch_pair_prices,
    fetch_company_metrics,
    fetch_earnings_history,
    compute_fundamental_divergence,
    fetch_ticker_news,
)
from src.features import (
    build_stats_arb_feature_set,
    compute_gap_classification,
    compute_sentiment_gap,
    calculate_support_resistance_channels,
)
from src.ml import (
    train_deep_gap_model,
    run_comparative_backtest,
)
from src.ai import generate_ai_trade_summary
from src.ui import (
    plot_normalized_prices,
    plot_spread_and_bands,
    plot_zscore_timeline,
    plot_support_resistance_channels,
    plot_backtest_equity_curves,
    plot_dl_loss_curves,
    plot_feature_importance,
    render_header_kpis,
    render_fundamental_table,
    render_ai_memo_card,
    render_news_feed,
)

# Page configuration
st.set_page_config(
    page_title="Pairs Trading & Fundamental Gap Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 AI Pairs Trading & Fundamental Gap Analyzer")
st.caption("Quantitative Statistical Arbitrage, Deep Learning Signal Generation & Fundamental Relative Valuation")

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. Preset or Custom Pair
    pair_mode = st.radio("Pair Selection Mode", ["Preset Peers", "Custom Tickers"], horizontal=True)
    
    if pair_mode == "Preset Peers":
        preset_choice = st.selectbox("Select Peer Group", list(PEER_PRESETS.keys()), index=0)
        ticker_a, ticker_b = PEER_PRESETS[preset_choice]
    else:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            ticker_a = st.text_input("Ticker A", value="KO").upper().strip()
        with col_t2:
            ticker_b = st.text_input("Ticker B", value="PEP").upper().strip()
            
    st.divider()
    
    # 2. Market Parameters
    st.subheader("Data & Lookback")
    lookback_options = ["1y", "2y", "3y", "5y"]
    lookback = st.selectbox(
        "Historical Lookback",
        lookback_options,
        index=lookback_options.index(DEFAULT_LOOKBACK_PERIOD),
    )
    rolling_window = st.slider("Rolling Window (Days)", min_value=15, max_value=90, value=DEFAULT_ROLLING_WINDOW, step=5)
    entry_z = st.slider("Z-Score Entry Threshold (σ)", min_value=1.0, max_value=3.0, value=DEFAULT_ENTRY_Z, step=0.25)
    
    st.divider()
    
    # 3. OpenRouter Free AI
    st.subheader("🤖 OpenRouter AI")
    api_key = st.text_input("OpenRouter API Key (Optional)", type="password", help="Enter key to use free AI models, or leave blank to use the built-in deterministic quantitative engine.")
    selected_model = st.selectbox(
        "Model", OPENROUTER_FREE_MODELS, index=OPENROUTER_FREE_MODELS.index(DEFAULT_MODEL)
    )
    
    st.divider()
    
    # 4. Deep Learning Settings
    with st.expander("🧠 Deep Learning Settings"):
        dl_epochs = st.slider("Epochs", 10, 100, DEFAULT_DL_EPOCHS, 5)
        dl_lr = st.select_slider("Learning Rate", options=[0.001, 0.005, 0.01, 0.02], value=DEFAULT_DL_LR)
        dl_horizon = st.slider("Forecast Horizon (Days)", 3, 20, DEFAULT_DL_PRED_HORIZON, 1)

    with st.expander("📐 Support / Resistance Channels"):
        sr_pivot_period = st.slider("Pivot Period", 4, 30, 10)
        sr_source = st.selectbox("Pivot Source", ["High/Low", "Close/Open"])
        sr_channel_width = st.slider("Maximum Channel Width (%)", 1, 8, 5)
        sr_min_strength = st.slider("Minimum Pivot Count", 1, 10, 1)
        sr_max_channels = st.slider("Maximum Channels", 1, 10, 6)
        sr_loopback = st.slider("Loopback Period", 100, 400, 290, step=10)
        sr_show_pivots = st.checkbox("Show Confirmed Pivot Points", value=False)
        sr_show_breakouts = st.checkbox("Mark Latest Channel Breakouts", value=False)
        sr_show_ma1 = st.checkbox("Show Moving Average 1", value=False)
        sr_ma1_length = st.number_input("MA 1 Length", min_value=1, max_value=500, value=50)
        sr_ma1_type = st.selectbox("MA 1 Type", ["SMA", "EMA"], key="sr_ma1_type")
        sr_show_ma2 = st.checkbox("Show Moving Average 2", value=False)
        sr_ma2_length = st.number_input("MA 2 Length", min_value=1, max_value=500, value=200)
        sr_ma2_type = st.selectbox("MA 2 Type", ["SMA", "EMA"], key="sr_ma2_type")

    refresh_btn = st.button("🔄 Refresh Data & Re-Analyze", type="primary", use_container_width=True)

# ----------------- CACHED DATA PIPELINE -----------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_all_pair_data(sym_a: str, sym_b: str, period: str, window: int):
    # 1. Prices
    price_df, meta = fetch_pair_prices(sym_a, sym_b, period=period)
    # 2. Features
    feat_df, stats_meta = build_stats_arb_feature_set(price_df, sym_a, sym_b, window=window)
    # 3. Fundamentals
    metrics_a = fetch_company_metrics(sym_a)
    metrics_b = fetch_company_metrics(sym_b)
    fund_diff = compute_fundamental_divergence(metrics_a, metrics_b)
    earnings_a = fetch_earnings_history(sym_a).to_dict(orient="records")
    earnings_b = fetch_earnings_history(sym_b).to_dict(orient="records")
    # 4. News
    news_a = fetch_ticker_news(sym_a, limit=6)
    news_b = fetch_ticker_news(sym_b, limit=6)
    sentiment_gap = compute_sentiment_gap(news_a, news_b, sym_a, sym_b)
    
    return {
        "price_df": price_df,
        "feat_df": feat_df,
        "meta": meta,
        "stats_meta": stats_meta,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "fund_diff": fund_diff,
        "earnings_a": earnings_a,
        "earnings_b": earnings_b,
        "news_a": news_a,
        "news_b": news_b,
        "sentiment_gap": sentiment_gap,
    }

if refresh_btn:
    load_all_pair_data.clear()
    st.rerun()

# Load data with loading spinner
try:
    with st.spinner(f"Fetching financial market data & fundamentals for {ticker_a} and {ticker_b}..."):
        bundle = load_all_pair_data(ticker_a, ticker_b, lookback, rolling_window)
except Exception as e:
    st.error(f"Failed to load pair data: {e}")
    st.stop()

analysis_context = (ticker_a, ticker_b, lookback, rolling_window)
if st.session_state.get("analysis_context") != analysis_context:
    for key in (
        "ai_memo_result",
        "ai_memo_context",
        "dl_model",
        "dl_metrics",
        "dl_signals",
        "dl_importances",
        "dl_split_date",
    ):
        st.session_state.pop(key, None)
    st.session_state["analysis_context"] = analysis_context

memo_context = (
    analysis_context,
    selected_model,
    hashlib.sha256(api_key.encode("utf-8")).hexdigest() if api_key else "",
)
if st.session_state.get("ai_memo_context") != memo_context:
    st.session_state.pop("ai_memo_result", None)
    st.session_state["ai_memo_context"] = memo_context

dl_context = (analysis_context, dl_epochs, dl_lr, dl_horizon)
if st.session_state.get("dl_context") != dl_context:
    for key in ("dl_model", "dl_metrics", "dl_signals", "dl_importances", "dl_split_date"):
        st.session_state.pop(key, None)
    st.session_state["dl_context"] = dl_context

# Unpack data bundle
feat_df = bundle["feat_df"]
meta = bundle["meta"]
stats_meta = bundle["stats_meta"]
metrics_a = bundle["metrics_a"]
metrics_b = bundle["metrics_b"]
fund_diff = bundle["fund_diff"]
earnings_a = bundle["earnings_a"]
earnings_b = bundle["earnings_b"]
news_a = bundle["news_a"]
news_b = bundle["news_b"]
sentiment_stats = bundle["sentiment_gap"]

# Calculate Gap Classification
gap_classification = compute_gap_classification(
    z_score=stats_meta["current_zscore"],
    fundamental_divergence=fund_diff,
    coint_meta=stats_meta,
    threshold_z=entry_z,
)

# ----------------- TOP HEADER KPIS -----------------
render_header_kpis(meta[ticker_a], meta[ticker_b], stats_meta, gap_classification)

st.markdown("---")

# ----------------- MAIN INTERFACE TABS -----------------
tab_overview, tab_fundamentals, tab_ai, tab_dl, tab_backtest, tab_news = st.tabs([
    "📊 Gap Visualizer",
    "🏢 Fundamentals & Earnings",
    "🤖 AI Trade Memo",
    "🧠 Deep Learning Signal Engine",
    "📈 Backtest & Strategy Comparison",
    "📰 News & Sentiment",
])

# ----- TAB 1: GAP VISUALIZER -----
with tab_overview:
    st.subheader(f"Price Spread Dislocation & Statistical Gap ({ticker_a} vs {ticker_b})")
    
    # 1. Spread and Bollinger Bands with Gap Zones
    fig_spread = plot_spread_and_bands(feat_df, ticker_a, ticker_b)
    st.plotly_chart(fig_spread, use_container_width=True)
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        # Normalized Prices
        fig_norm = plot_normalized_prices(feat_df, ticker_a, ticker_b)
        st.plotly_chart(fig_norm, use_container_width=True)
    with col_chart2:
        # Rolling Z-score
        fig_z = plot_zscore_timeline(feat_df)
        st.plotly_chart(fig_z, use_container_width=True)

    st.markdown("#### Support / Resistance Channels")
    sr_ticker = st.selectbox("Channel chart ticker", [ticker_a, ticker_b], key="sr_ticker")
    sr_channels, sr_pivots = calculate_support_resistance_channels(
        price_df=bundle["price_df"],
        ticker=sr_ticker,
        pivot_period=sr_pivot_period,
        source=sr_source,
        channel_width_pct=sr_channel_width,
        min_strength=sr_min_strength,
        max_channels=sr_max_channels,
        loopback=sr_loopback,
    )
    sr_moving_averages = []
    if sr_show_ma1:
        sr_moving_averages.append(
            {"length": sr_ma1_length, "type": sr_ma1_type, "color": "#42a5f5"}
        )
    if sr_show_ma2:
        sr_moving_averages.append(
            {"length": sr_ma2_length, "type": sr_ma2_type, "color": "#ef5350"}
        )
    sr_chart = plot_support_resistance_channels(
        bundle["price_df"],
        sr_ticker,
        sr_channels,
        sr_pivots,
        show_pivots=sr_show_pivots,
        show_breakouts=sr_show_breakouts,
        moving_averages=sr_moving_averages,
    )
    st.plotly_chart(sr_chart, use_container_width=True)
    if not sr_channels:
        st.info("No support/resistance channels meet the selected pivot and strength settings.")
    else:
        st.caption(
            f"Showing {len(sr_channels)} strongest channels from confirmed pivots. "
            "Channel levels are descriptive and are not used as pairs-trading signals."
        )
        
    # Statistical Arbitrage Diagnostics Card
    with st.expander("🔍 Detailed Statistical Arbitrage Diagnostics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.write(f"**Hedge Ratio (β)**: `{stats_meta['beta']:.4f}`")
        col1.write(f"**OLS Intercept (α)**: `{stats_meta['alpha']:.4f}`")
        col2.write(f"**Cointegration p-value**: `{stats_meta['coint_p_value']:.4f}`")
        col2.write(f"**Statistically Cointegrated**: `{stats_meta['is_cointegrated']}`")
        col3.write(f"**Half-Life of Reversion**: `{stats_meta['half_life_days']:.1f} days`")
        col3.write(f"**Hurst Exponent**: `{stats_meta['hurst_exponent']:.3f}`")
        col4.write(f"**Market Regime**: `{stats_meta['regime']}`")
        col4.write(f"**Current Z-Score**: `{stats_meta['current_zscore']:+.2f}σ`")

# ----- TAB 2: FUNDAMENTALS & EARNINGS -----
with tab_fundamentals:
    st.subheader(f"Relative Fundamental Matrix: {ticker_a} vs {ticker_b}")
    render_fundamental_table(metrics_a, metrics_b, fund_diff, earnings_a, earnings_b)
    
    st.info(
        "💡 **Key Quant Concept**: If a stock pair shows a price gap (|Z| ≥ 2.0), but their margins, "
        "earnings surprises, and forward P/E ratios are performing at parity, it is an **Unwarranted Statistical Dislocation** "
        "(Buy Signal). If one peer missed earnings or lowered guidance, the gap is **Fundamentally Justified** (Do Not Trade)."
    )

# ----- TAB 3: AI TRADE MEMO -----
with tab_ai:
    st.subheader("🤖 Institutional AI Investment Memo & Gap Synthesis")
    
    col_ai_btn, col_ai_info = st.columns([1, 3])
    with col_ai_btn:
        generate_memo_btn = st.button("✨ Generate AI Investment Brief", type="primary", use_container_width=True)
        
    if generate_memo_btn or "ai_memo_result" not in st.session_state:
        with st.spinner("Synthesizing quantitative metrics, quarterly surprises, and news catalysts..."):
            ai_res = generate_ai_trade_summary(
                ticker_a=ticker_a,
                ticker_b=ticker_b,
                meta_a=meta[ticker_a],
                meta_b=meta[ticker_b],
                stats_meta=stats_meta,
                metrics_a=metrics_a,
                metrics_b=metrics_b,
                fundamental_diff=fund_diff,
                earnings_a=earnings_a,
                earnings_b=earnings_b,
                news_a=news_a,
                news_b=news_b,
                sentiment_stats=sentiment_stats,
                api_key=api_key,
                model_name=selected_model,
            )
            st.session_state["ai_memo_result"] = ai_res
            
    render_ai_memo_card(st.session_state["ai_memo_result"], gap_classification)

# ----- TAB 4: DEEP LEARNING SIGNAL ENGINE -----
with tab_dl:
    st.subheader("🧠 Deep Learning Signal Engine (DeepGapNet)")
    st.markdown(
        "A Deep Neural Network with Batch Normalization, Dropout (0.2), and LeakyReLU activations "
        "trained to predict forward spread mean-reversion magnitude and generate non-linear trading signals."
    )
    
    col_train, col_status = st.columns([1, 2])
    with col_train:
        train_dl_btn = st.button("🚀 Train Deep Learning Model", type="primary", use_container_width=True)
        
    if train_dl_btn or "dl_signals" not in st.session_state:
        with st.spinner(f"Training DeepGapNet for {dl_epochs} epochs with Huber loss & Adam optimizer..."):
            res = train_deep_gap_model(
                feat_df=feat_df,
                ticker_a=ticker_a,
                ticker_b=ticker_b,
                fundamental_diff=fund_diff,
                sentiment_gap=sentiment_stats.get("sentiment_gap", 0.0),
                pred_horizon=dl_horizon,
                epochs=dl_epochs,
                batch_size=DEFAULT_DL_BATCH_SIZE,
                learning_rate=dl_lr,
            )
            if len(res) == 5:
                model, train_metrics, dl_signals, feat_importances, split_date = res
            else:
                model, train_metrics, dl_signals, feat_importances = res
                split_date = feat_df.index[int(len(feat_df) * 0.8)]
                
            st.session_state["dl_model"] = model
            st.session_state["dl_metrics"] = train_metrics
            st.session_state["dl_signals"] = dl_signals
            st.session_state["dl_importances"] = feat_importances
            st.session_state["dl_split_date"] = split_date
            st.success(f"Model successfully trained! Final Train Loss: {train_metrics['final_train_loss']:.5f} | Best Val Loss: {train_metrics['final_val_loss']:.5f}")

    if "dl_metrics" in st.session_state:
        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            fig_loss = plot_dl_loss_curves(
                st.session_state["dl_metrics"]["train_losses"],
                st.session_state["dl_metrics"]["val_losses"],
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        with col_fig2:
            fig_imp = plot_feature_importance(st.session_state["dl_importances"])
            st.plotly_chart(fig_imp, use_container_width=True)
            
        # Display current Deep Learning Signal
        latest_dl_signal = st.session_state["dl_signals"][-1]
        sig_map = {1: "🟢 LONG SPREAD (Buy A / Short B)", -1: "🔴 SHORT SPREAD (Short A / Buy B)", 0: "⚪ NEUTRAL / FLAT"}
        st.markdown(f"### Current Deep Learning Prediction: **{sig_map.get(latest_dl_signal, 'NEUTRAL')}**")

# ----- TAB 5: BACKTESTING & STRATEGY COMPARISON -----
with tab_backtest:
    st.subheader("📈 Backtest & Strategy Comparison")
    st.caption("Comparing Standard Statistical Z-Score, Fundamentally-Filtered Z-Score, and the Deep Learning Signal Engine.")
    
    # Run comparative backtest with split date
    dl_signals_array = st.session_state.get("dl_signals")
    split_date = st.session_state.get("dl_split_date")
    
    backtest_results = run_comparative_backtest(
        feat_df=feat_df,
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        beta=stats_meta["beta"],
        dl_signals=dl_signals_array,
        split_date=split_date,
        entry_z=entry_z,
        exit_z=DEFAULT_EXIT_Z,
        stop_loss_z=DEFAULT_STOP_LOSS_Z,
    )
    
    # 1. Equity Curves Plot with Train/Test Demarcation
    fig_equity = plot_backtest_equity_curves(backtest_results, split_date=split_date)
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # 2. In-Sample vs Out-of-Sample Transparency Card
    dl_res = backtest_results.get("deep_learning")
    if dl_res and dl_res.get("out_of_sample_metrics"):
        is_m = dl_res["in_sample_metrics"]
        oos_m = dl_res["out_of_sample_metrics"]
        st.markdown(f"#### 🔬 In-Sample (Train) vs. Out-of-Sample (Test) Validation Breakdown (Cutoff: `{str(split_date)[:10]}`)")
        
        col_is, col_oos = st.columns(2)
        with col_is:
            st.info(f"""
            **In-Sample (Training Period)**
            - **Total Return**: `{is_m['total_return_pct']:+.2f}%`
            - **CAGR**: `{is_m['cagr_pct']:+.2f}%`
            - **Sharpe Ratio**: `{is_m['sharpe_ratio']:.2f}`
            - **Max Drawdown**: `{is_m['max_drawdown_pct']:.2f}%`
            - **Win Rate**: `{is_m['win_rate_pct']:.1f}%` ({is_m['total_trades']} trades)
            """)
        with col_oos:
            st.success(f"""
            **Out-of-Sample (Unseen Test Period)**
            - **Total Return**: `{oos_m['total_return_pct']:+.2f}%`
            - **CAGR**: `{oos_m['cagr_pct']:+.2f}%`
            - **Sharpe Ratio**: `{oos_m['sharpe_ratio']:.2f}`
            - **Max Drawdown**: `{oos_m['max_drawdown_pct']:.2f}%`
            - **Win Rate**: `{oos_m['win_rate_pct']:.1f}%` ({oos_m['total_trades']} trades)
            """)

    # 3. Overall Performance Comparison Table
    st.markdown("#### Overall Period Performance Matrix ($100,000 Starting Capital)")
    perf_rows = []
    for key, res in backtest_results.items():
        if isinstance(res, dict) and "metrics" in res:
            m = res["metrics"]
            perf_rows.append({
                "Strategy": m["strategy_name"],
                "Total Return (%)": f"{m['total_return_pct']:+.2f}%",
                "CAGR (%)": f"{m['cagr_pct']:+.2f}%",
                "Sharpe Ratio": f"{m['sharpe_ratio']:.2f}",
                "Sortino Ratio": f"{m['sortino_ratio']:.2f}",
                "Max Drawdown (%)": f"{m['max_drawdown_pct']:.2f}%",
                "Win Rate (%)": f"{m['win_rate_pct']:.1f}%",
                "Total Trades": m["total_trades"],
                "Profit Factor": f"{m['profit_factor']:.2f}",
            })
    st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)
    
    # 3. Trade Log
    with st.expander("📜 Recent Completed Trade Log (Statistical Strategy)"):
        trades_list = backtest_results["statistical"]["trades"]
        if trades_list:
            st.dataframe(pd.DataFrame(trades_list).tail(15), use_container_width=True, hide_index=True)
        else:
            st.info("No trades executed within the selected parameters.")

# ----- TAB 6: NEWS STREAM & SENTIMENT -----
with tab_news:
    st.subheader("📰 Real-Time Company News Stream & NLP Sentiment")
    render_news_feed(news_a, news_b, ticker_a, ticker_b)
    
    st.markdown(f"**Relative Sentiment Gap**: `{sentiment_stats['sentiment_gap']:+.2f}` — {sentiment_stats['gap_label']}")
