# Portfolio Lab

Constrained mean–variance optimization with minimum-volatility, maximum-Sharpe, and target-return portfolios; efficient frontier; custom weights; sensitivity; correlations; drawdown; and a chronological holdout backtest.

The default price history is synthetic. Upload a wide CSV whose first column is a date and whose remaining columns are asset prices.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Historical estimates are uncertain. Results omit fees, taxes, liquidity, and changing market structure and are not investment advice.

