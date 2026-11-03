# 📈 AI-Powered Pairs Trading & Fundamental Gap Analyzer

A quantitative relative-valuation and pairs trading platform designed to identify, analyze, backtest, and synthesize trading opportunities between peer stocks (e.g., Coca-Cola `KO` vs. PepsiCo `PEP`).

The system determines whether an observed price gap represents an **unwarranted statistical dislocation (Mean-Reversion / Buy Signal)** or a **justified fundamental divergence (Do-Not-Fade / Fundamental Shift)** using quantitative feature engineering, a Deep Neural Network trading signal engine (`DeepGapNet`), news sentiment, and OpenRouter AI synthesis.

---

## 🚀 Key Highlights & Architecture

- **Clean Modular Pipeline (Zero LangGraph Overhead)**:
  - `src/data/`: High-performance ingestion for prices, financial ratios, quarterly earnings surprises, and news feeds.
  - `src/features/`: Dynamic OLS hedge ratio ($\beta$), rolling Z-score ($\pm 1\sigma, \pm 2\sigma$), Engle-Granger cointegration test, Ornstein-Uhlenbeck half-life, Hurst exponent, and fundamental disparity index.
    - Pivot-based support/resistance channels on either pair ticker, with confirmed pivot markers, configurable strength/loopback, optional SMA/EMA overlays, and latest-bar breakout markers.
  - `src/ml/`:
    - **`DeepGapNet`**: NumPy neural network with Batch Normalization, Dropout, LeakyReLU activations, and mini-batch Adam optimization predicting forward spread change. Its fit/validation/test ranges are chronological and purged by the forecast horizon; current fundamental/news snapshots are excluded from historical model features unless timestamped observations are supplied.
    - **Vectorized Backtester**: 3-way comparative backtester (Statistical Z-Score vs. Fundamental Filter vs. Deep Learning Model) simulating dollar-neutral pairs positions with transaction costs and slippage.
  - `src/ai/`: Direct OpenRouter client with free models (`google/gemini-2.0-flash-exp:free`, `meta-llama/llama-3.3-70b-instruct:free`, `qwen/qwen-2.5-72b-instruct:free`), backed by an intelligent offline quantitative heuristic engine.
  - `src/ui/`: Interactive Plotly visual charts and Streamlit components.

---

## 💻 Quick Start & Running the App

### 1. Launch the Streamlit Interface
Run the following command in your terminal from the project root:

```powershell
python -m streamlit run app.py
```

### 2. Run Automated Unit & Integration Tests
```powershell
python run_tests.py
```

### 3. Run Live End-to-End Pipeline Verification
```powershell
python test_live_pipeline.py
```

---

## 📊 Dashboard Capabilities

1. **📊 Gap Visualizer**:
   - Normalized price comparison (Base 100).
   - OLS residual price spread with rolling mean and $\pm 1\sigma, \pm 2\sigma$ Bollinger divergence bands.
   - Shaded visual markers indicating overbought and oversold potential gap zones.
   - Cointegration p-value, mean-reverting half-life (days), and Hurst exponent regime.

2. **🏢 Fundamentals & Earnings Peer Matrix**:
   - Side-by-side comparison of Forward P/E, EV/EBITDA, Operating Margins, YoY Revenue Growth, ROE, and Dividend Yield.
   - Latest quarterly reported EPS vs. analyst consensus estimates and surprise percentages.
   - Composite Fundamental Parity status.

3. **🤖 AI Trade Memo**:
   - Institutional investment memo synthesizing quantitative metrics, quarterly surprises, and news catalysts.
   - Definitive trade actions (`LONG PAIR`, `SHORT PAIR`, or `AVOID REVERSION`).

4. **🧠 Deep Learning Signal Engine (`DeepGapNet`)**:
   - Interactive model training with custom epochs, learning rate, and forecast horizon.
   - Epoch-by-epoch Training and Validation Huber Loss curves.
   - Permutation Feature Importance for trailing price, spread, momentum, and volatility features. Current fundamental/news snapshots are not projected backward across historical data, avoiding point-in-time leakage.

5. **📈 Backtesting & Strategy Comparison**:
   - Head-to-head simulation of:
     1. Baseline Statistical Bollinger Z-Score Strategy
     2. Fundamentally-Filtered Z-Score Strategy
     3. Deep Learning Neural Network Trading Signal Strategy
     4. Buy & Hold 50/50 Benchmark
   - Metrics: Total Return, CAGR, Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Win Rate, and full trade log.

6. **📰 News Stream & Sentiment**:
   - Real-time headline feed with NLP sentiment tags (Bullish / Bearish / Neutral) and relative sentiment gap.

7. **📐 Support / Resistance Channels**:
   - OHLC chart for either selected ticker, with channel bands clustered from confirmed pivot highs/lows.
   - Configurable pivot period, pivot source, channel width, pivot-count threshold, loopback, and maximum displayed channels.
   - Optional confirmed pivot markers, latest-bar breakout markers, and two SMA/EMA overlays.
   - These levels are technical context only and do not feed the pairs strategy or backtest.

The support/resistance channel approach is based on the user-provided TradingView script by LonesomeTheBlue, published under the Mozilla Public License 2.0: https://mozilla.org/MPL/2.0/.
