"""Streamlit visual UI components, metrics cards, tables, and AI banners."""

from typing import Any, Dict, List
import pandas as pd
import streamlit as st

def render_header_kpis(
    meta_a: Dict[str, Any],
    meta_b: Dict[str, Any],
    stats_meta: Dict[str, Any],
    gap_class: Dict[str, Any]
):
    """Render top summary metrics and current gap status."""
    sym_a = meta_a.get("short_name", "Stock A")
    sym_b = meta_b.get("short_name", "Stock B")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label=f"{sym_a}",
            value=f"${meta_a.get('current_price', 0.0):.2f}",
            delta=f"{meta_a.get('change_pct', 0.0):+.2f}%",
        )
    with col2:
        st.metric(
            label=f"{sym_b}",
            value=f"${meta_b.get('current_price', 0.0):.2f}",
            delta=f"{meta_b.get('change_pct', 0.0):+.2f}%",
        )
    with col3:
        z = stats_meta.get("current_zscore", 0.0)
        st.metric(
            label="Spread Z-Score",
            value=f"{z:+.2f} σ",
            delta="Overbought" if z >= 2.0 else ("Oversold" if z <= -2.0 else "Normal"),
            delta_color="inverse" if abs(z) >= 2.0 else "off",
        )
    with col4:
        coint_text = "Passed (p<0.05)" if stats_meta.get("is_cointegrated") else "Not Coint."
        st.metric(
            label="Cointegration",
            value=coint_text,
            delta=f"p={stats_meta.get('coint_p_value', 0.5):.3f}",
        )
    with col5:
        st.metric(
            label="Mean-Rev Half-Life",
            value=f"{stats_meta.get('half_life_days', 0):.1f} d",
            delta=stats_meta.get("regime", "Regime"),
        )
        
    # Prominent Signal Banner
    signal = gap_class.get("signal", "NEUTRAL")
    gap_type = gap_class.get("gap_type", "IN_RANGE")
    
    if signal == "LONG_PAIR":
        st.success(f"🟢 **SIGNAL: LONG PAIR** (Buy {meta_a.get('short_name')} / Short {meta_b.get('short_name')}) — {gap_class.get('recommendation')}")
    elif signal == "SHORT_PAIR":
        st.warning(f"🔴 **SIGNAL: SHORT PAIR** (Short {meta_a.get('short_name')} / Buy {meta_b.get('short_name')}) — {gap_class.get('recommendation')}")
    elif signal == "AVOID_REVERSION":
        st.error(f"⛔ **SIGNAL: AVOID REVERSION** — Justified Structural Divergence. {gap_class.get('recommendation')}")
    else:
        st.info(f"⚪ **SIGNAL: NEUTRAL** — {gap_class.get('recommendation')}")

def render_fundamental_table(
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
    fundamental_diff: Dict[str, Any],
    earnings_a: List[Dict[str, Any]],
    earnings_b: List[Dict[str, Any]]
):
    """Render comprehensive fundamental peer comparison matrix."""
    sym_a = metrics_a.get("ticker", "A")
    sym_b = metrics_b.get("ticker", "B")
    
    # Valuation & Operational Metrics
    data = [
        {"Metric": "Forward P/E", sym_a: metrics_a.get("forward_pe", "N/A"), sym_b: metrics_b.get("forward_pe", "N/A"), "Spread / Delta": fundamental_diff.get("pe_diff", "N/A")},
        {"Metric": "Trailing P/E", sym_a: metrics_a.get("trailing_pe", "N/A"), sym_b: metrics_b.get("trailing_pe", "N/A"), "Spread / Delta": "-"},
        {"Metric": "EV / EBITDA", sym_a: metrics_a.get("ev_to_ebitda", "N/A"), sym_b: metrics_b.get("ev_to_ebitda", "N/A"), "Spread / Delta": fundamental_diff.get("ev_ebitda_diff", "N/A")},
        {"Metric": "Operating Margin", sym_a: f"{metrics_a.get('operating_margin', 0)*100:.2f}%" if metrics_a.get('operating_margin') else "N/A", sym_b: f"{metrics_b.get('operating_margin', 0)*100:.2f}%" if metrics_b.get('operating_margin') else "N/A", "Spread / Delta": f"{fundamental_diff.get('op_margin_diff', 0)*100:+.2f}%" if fundamental_diff.get('op_margin_diff') else "N/A"},
        {"Metric": "YoY Revenue Growth", sym_a: f"{metrics_a.get('revenue_growth', 0)*100:.2f}%" if metrics_a.get('revenue_growth') else "N/A", sym_b: f"{metrics_b.get('revenue_growth', 0)*100:.2f}%" if metrics_b.get('revenue_growth') else "N/A", "Spread / Delta": f"{fundamental_diff.get('rev_growth_diff', 0)*100:+.2f}%" if fundamental_diff.get('rev_growth_diff') else "N/A"},
        {"Metric": "Return on Equity (ROE)", sym_a: f"{metrics_a.get('return_on_equity', 0)*100:.2f}%" if metrics_a.get('return_on_equity') else "N/A", sym_b: f"{metrics_b.get('return_on_equity', 0)*100:.2f}%" if metrics_b.get('return_on_equity') else "N/A", "Spread / Delta": f"{fundamental_diff.get('roe_diff', 0)*100:+.2f}%" if fundamental_diff.get('roe_diff') else "N/A"},
        {"Metric": "Dividend Yield", sym_a: f"{metrics_a.get('dividend_yield', 0)*100:.2f}%" if metrics_a.get('dividend_yield') else "N/A", sym_b: f"{metrics_b.get('dividend_yield', 0)*100:.2f}%" if metrics_b.get('dividend_yield') else "N/A", "Spread / Delta": "-"},
    ]
    df_metrics = pd.DataFrame(data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    
    st.markdown(f"**Composite Fundamental Parity Status**: `{fundamental_diff.get('parity_status')}` (Parity Score: `{fundamental_diff.get('parity_score', 0):+.4f}`)")
    
    # Quarterly Earnings Surprise comparison
    st.markdown("#### Latest Reported Earnings vs Consensus Estimates")
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"**{sym_a} Earnings History**")
        if earnings_a:
            st.dataframe(pd.DataFrame(earnings_a).head(4), use_container_width=True, hide_index=True)
        else:
            st.info("No earnings surprise history available.")
    with col2:
        st.caption(f"**{sym_b} Earnings History**")
        if earnings_b:
            st.dataframe(pd.DataFrame(earnings_b).head(4), use_container_width=True, hide_index=True)
        else:
            st.info("No earnings surprise history available.")

def render_ai_memo_card(ai_result: Dict[str, Any], gap_class: Dict[str, Any]):
    """Render the AI investment memo card."""
    source = ai_result.get("source", "AI Engine")
    memo = ai_result.get("memo", "")
    
    st.markdown(f"""
    <div style="background-color: #1e2638; padding: 18px; border-radius: 8px; border-left: 5px solid #00e5ff; margin-bottom: 20px;">
        <span style="background-color: #0d47a1; color: white; padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;">
            Source: {source}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(memo)

def render_news_feed(
    news_a: List[Dict[str, Any]],
    news_b: List[Dict[str, Any]],
    ticker_a: str,
    ticker_b: str
):
    """Render news stream with sentiment labels."""
    col1, col2 = st.columns(2)
    
    def render_articles(items, sym):
        if not items:
            st.caption(f"No recent news found for {sym}.")
            return
        for item in items:
            s = item.get("sentiment_score", 0.0)
            badge_color = "#4caf50" if s > 0.15 else ("#f44336" if s < -0.15 else "#9e9e9e")
            badge_text = "Bullish" if s > 0.15 else ("Bearish" if s < -0.15 else "Neutral")
            
            st.markdown(f"""
            <div style="padding: 10px; margin-bottom: 8px; background-color: #1a1f2c; border-radius: 6px;">
                <span style="font-size: 11px; color: {badge_color}; font-weight: bold; border: 1px solid {badge_color}; padding: 1px 5px; border-radius: 3px;">
                    {badge_text} ({s:+.2f})
                </span>
                <span style="font-size: 11px; color: #888; margin-left: 8px;">{item.get('publish_date')} | {item.get('publisher')}</span>
                <div style="margin-top: 5px; font-size: 13px; font-weight: 500;">
                    <a href="{item.get('link')}" target="_blank" style="color: #64b5f6; text-decoration: none;">{item.get('title')}</a>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col1:
        st.subheader(f"{ticker_a} News Digest")
        render_articles(news_a, ticker_a)
        
    with col2:
        st.subheader(f"{ticker_b} News Digest")
        render_articles(news_b, ticker_b)

def render_backtest_kpi_cards(metrics: Dict[str, Any]):
    """Render backtest performance cards."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        ret = metrics.get("total_return_pct", 0.0)
        st.metric("Total Return", f"{ret:+.2f}%")
    with col2:
        st.metric("CAGR", f"{metrics.get('cagr_pct', 0.0):+.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0.0):.2f}")
    with col4:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0.0):.2f}%")
    with col5:
        st.metric("Win Rate", f"{metrics.get('win_rate_pct', 0.0):.1f}%", f"{metrics.get('total_trades', 0)} Trades")
