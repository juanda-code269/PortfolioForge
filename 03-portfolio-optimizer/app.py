import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(page_title="Portfolio Lab", page_icon="📈", layout="wide")
st.title("Portfolio Lab")
st.caption("Efficient-frontier and allocation experiments — educational, not investment advice.")


@st.cache_data
def demo_prices():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2019-01-01", periods=1250)
    names = ["US Equity", "Intl Equity", "Bonds", "REITs", "Gold", "Small Cap"]
    annual_mu = np.array([.09, .075, .035, .065, .045, .095])
    annual_vol = np.array([.18, .20, .065, .22, .16, .25])
    corr = np.array([
        [1,.78,-.18,.62,.05,.84],[.78,1,-.12,.58,.12,.72],[-.18,-.12,1,-.08,.18,-.22],
        [.62,.58,-.08,1,.09,.60],[.05,.12,.18,.09,1,.02],[.84,.72,-.22,.60,.02,1]
    ])
    cov = np.outer(annual_vol, annual_vol) * corr / 252
    daily = rng.multivariate_normal(annual_mu / 252, cov, len(dates))
    return pd.DataFrame(100 * np.exp(np.cumsum(daily, axis=0)), index=dates, columns=names)


def read_prices(upload):
    if upload is None:
        return demo_prices(), "Synthetic demonstration series"
    raw = pd.read_csv(upload)
    date_col = next((c for c in raw.columns if c.lower() in {"date", "time", "timestamp"}), raw.columns[0])
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    frame = raw.set_index(date_col).select_dtypes("number").sort_index().dropna(how="all")
    if frame.shape[1] < 2 or len(frame) < 30:
        raise ValueError("Upload needs a date column, at least two numeric price columns, and 30 rows.")
    return frame.ffill().dropna(), f"Uploaded file: {upload.name}"


def stats(weights, mu, cov, rf):
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    return ret, vol, (ret - rf) / vol if vol else np.nan


def solve(mu, cov, rf, lower, upper, objective, target=None):
    n = len(mu)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target is not None:
        constraints.append({"type": "eq", "fun": lambda w: w @ mu - target})
    if objective == "vol":
        fn = lambda w: np.sqrt(w @ cov @ w)
    else:
        fn = lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w)
    result = minimize(fn, np.repeat(1/n, n), bounds=[(lower, upper)] * n,
                      constraints=constraints, method="SLSQP", options={"maxiter": 1000})
    return result.x if result.success else None


with st.sidebar:
    upload = st.file_uploader("Wide price CSV", type="csv", help="First column: date; remaining columns: asset prices")
    rf = st.number_input("Risk-free rate", 0.0, .20, .03, .005, format="%.3f")
    min_w = st.slider("Minimum asset weight", 0.0, .20, 0.0, .01)
    max_w = st.slider("Maximum asset weight", .10, 1.0, .70, .05)
    shock = st.slider("Expected-return sensitivity shock", -.05, .05, 0.0, .005)

try:
    prices, source = read_prices(upload)
except Exception as exc:
    st.error(str(exc)); st.stop()

if min_w * prices.shape[1] > 1 or max_w * prices.shape[1] < 1:
    st.error("The selected allocation bounds are infeasible for this number of assets."); st.stop()

returns = prices.pct_change().dropna()
mu = returns.mean().to_numpy() * 252
mu[0] += shock
cov = returns.cov().to_numpy() * 252
assets = prices.columns.tolist()
min_vol = solve(mu, cov, rf, min_w, max_w, "vol")
max_sharpe = solve(mu, cov, rf, min_w, max_w, "sharpe")
if min_vol is None or max_sharpe is None:
    st.error("Optimization failed. Try wider allocation bounds or cleaner data."); st.stop()

tabs = st.tabs(["Overview", "Efficient frontier", "Custom portfolio", "Backtest", "Math & limitations"])
with tabs[0]:
    st.info(f"Data: {source}. Historical estimates are uncertain and are not forecasts.")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(prices / prices.iloc[0] * 100, title="Growth of 100"), width="stretch")
    c2.plotly_chart(px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Return correlations"), width="stretch")
    rows = []
    for label, w in [("Minimum volatility", min_vol), ("Maximum Sharpe", max_sharpe)]:
        r, v, s = stats(w, mu, cov, rf)
        rows.append({"Portfolio": label, "Expected return": r, "Volatility": v, "Sharpe": s})
    st.dataframe(pd.DataFrame(rows).style.format({"Expected return":"{:.1%}","Volatility":"{:.1%}","Sharpe":"{:.2f}"}), width="stretch")
    weights = pd.DataFrame({"Asset": assets, "Minimum volatility": min_vol, "Maximum Sharpe": max_sharpe}).melt("Asset", var_name="Portfolio", value_name="Weight")
    st.plotly_chart(px.bar(weights, x="Asset", y="Weight", color="Portfolio", barmode="group", title="Optimized allocations"), width="stretch")

with tabs[1]:
    low, high = stats(min_vol, mu, cov, rf)[0], max(mu)
    frontier = []
    for target in np.linspace(low, high, 45):
        w = solve(mu, cov, rf, min_w, max_w, "vol", target)
        if w is not None:
            r, v, s = stats(w, mu, cov, rf); frontier.append((v, r, s))
    ff = pd.DataFrame(frontier, columns=["Volatility", "Expected return", "Sharpe"])
    fig = px.scatter(ff, x="Volatility", y="Expected return", color="Sharpe", title="Constrained efficient frontier")
    for label, w in [("Min vol", min_vol), ("Max Sharpe", max_sharpe)]:
        r, v, _ = stats(w, mu, cov, rf); fig.add_scatter(x=[v], y=[r], name=label, marker_size=13)
    st.plotly_chart(fig, width="stretch")

with tabs[2]:
    st.write("Set relative weights; they are normalized to 100%.")
    raw = np.array([st.number_input(a, 0.0, 100.0, round(100/len(assets), 1), key=f"w_{a}") for a in assets])
    if raw.sum() == 0:
        st.warning("Enter at least one positive weight.")
    else:
        w = raw / raw.sum(); r, v, s = stats(w, mu, cov, rf)
        a, b, c = st.columns(3); a.metric("Expected return", f"{r:.1%}"); b.metric("Volatility", f"{v:.1%}"); c.metric("Sharpe", f"{s:.2f}")
        path = (1 + returns @ w).cumprod()
        drawdown = path / path.cummax() - 1
        st.metric("Maximum historical drawdown", f"{drawdown.min():.1%}")

with tabs[3]:
    cut = int(len(returns) * .7)
    train, test = returns.iloc[:cut], returns.iloc[cut:]
    w = solve(train.mean().to_numpy()*252, train.cov().to_numpy()*252, rf, min_w, max_w, "sharpe")
    if w is not None:
        out = pd.DataFrame({"Optimized on first 70%": (1 + test @ w).cumprod(), "Equal weight": (1 + test.mean(axis=1)).cumprod()})
        st.plotly_chart(px.line(out, title="Out-of-sample comparison (last 30%)"), width="stretch")
        st.caption("Weights are estimated only on the earlier period. This is a simple holdout, not evidence of future performance.")

with tabs[4]:
    st.markdown(r"""Portfolio variance is $w^T\Sigma w$: weights interact through covariance, so an asset can reduce total risk even when it is volatile on its own. The Sharpe objective is $(w^T\mu-r_f)/\sqrt{w^T\Sigma w}$.

**Important limitations.** Mean–variance optimization is highly sensitive to expected-return and covariance estimates. Results ignore taxes, fees, slippage, liquidity, and changing correlations. The sensitivity control deliberately shocks one estimate to expose this instability. Optimized does not mean guaranteed optimal.""")

