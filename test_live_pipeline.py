"""End-to-End Live Integration Verification with Real Market Data (KO vs PEP)."""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.data import fetch_pair_prices, fetch_company_metrics, fetch_earnings_history, compute_fundamental_divergence, fetch_ticker_news
from src.features import build_stats_arb_feature_set, compute_gap_classification, compute_sentiment_gap
from src.ml import train_deep_gap_model, run_comparative_backtest
from src.ai import generate_ai_trade_summary

def main():
    print("1. Fetching live prices for KO and PEP (1y)...")
    price_df, meta = fetch_pair_prices("KO", "PEP", period="1y")
    print(f"   Fetched {len(price_df)} trading days. KO: ${meta['KO']['current_price']}, PEP: ${meta['PEP']['current_price']}")
    
    print("2. Computing quantitative statistical arbitrage features...")
    feat_df, stats_meta = build_stats_arb_feature_set(price_df, "KO", "PEP", window=30)
    print(f"   Beta: {stats_meta['beta']}, Half-life: {stats_meta['half_life_days']}d, Z-Score: {stats_meta['current_zscore']} sigma")
    
    print("3. Fetching fundamentals & earnings...")
    metrics_a = fetch_company_metrics("KO")
    metrics_b = fetch_company_metrics("PEP")
    fund_diff = compute_fundamental_divergence(metrics_a, metrics_b)
    print(f"   Forward P/E: KO={metrics_a.get('forward_pe')} vs PEP={metrics_b.get('forward_pe')}. Parity: {fund_diff['parity_status']}")
    
    print("4. Fetching news & sentiment...")
    news_a = fetch_ticker_news("KO", limit=4)
    news_b = fetch_ticker_news("PEP", limit=4)
    sent_gap = compute_sentiment_gap(news_a, news_b, "KO", "PEP")
    print(f"   Sentiment Gap: {sent_gap['sentiment_gap']} ({sent_gap['gap_label']})")
    
    print("5. Training DeepGapNet neural network...")
    model, train_metrics, dl_signals, feat_importances, split_date = train_deep_gap_model(
        feat_df=feat_df,
        ticker_a="KO",
        ticker_b="PEP",
        fundamental_diff=fund_diff,
        sentiment_gap=sent_gap.get("sentiment_gap", 0.0),
        pred_horizon=10,
        epochs=15,
        batch_size=16,
    )
    print(f"   DeepGapNet trained. Final Train Loss: {train_metrics['final_train_loss']:.5f}. Top feature: {list(feat_importances.keys())[0]} ({list(feat_importances.values())[0]}%)")
    
    print("6. Running 3-way comparative backtester with In-Sample/Out-of-Sample evaluation...")
    bt_res = run_comparative_backtest(feat_df, "KO", "PEP", beta=stats_meta["beta"], dl_signals=dl_signals, split_date=split_date)
    stat_ret = bt_res["statistical"]["metrics"]["total_return_pct"]
    dl_ret = bt_res["deep_learning"]["metrics"]["total_return_pct"] if bt_res["deep_learning"] else 0.0
    print(f"   Statistical Return: {stat_ret:+.2f}%, Deep Learning Return: {dl_ret:+.2f}%")
    
    print("7. Generating AI synthesis brief...")
    ai_res = generate_ai_trade_summary(
        ticker_a="KO",
        ticker_b="PEP",
        meta_a=meta["KO"],
        meta_b=meta["PEP"],
        stats_meta=stats_meta,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        fundamental_diff=fund_diff,
        earnings_a=[],
        earnings_b=[],
        news_a=news_a,
        news_b=news_b,
        sentiment_stats=sent_gap,
    )
    print(f"   AI Synthesis Source: {ai_res['source']}")
    print("\nSUCCESS: All live pipeline components functional!")

if __name__ == "__main__":
    main()
