# app.py
# ---------------------------------------------------------
# Back-testing de carteras con interfaz Streamlit
# (funciona en Streamlit Community / Playground)
# ---------------------------------------------------------
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Backtest de Carteras", layout="wide")

# ------------- Funciones de negocio originales ----------------
@st.cache_data(show_spinner=False)
def download_daily_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, interval="1d", progress=False)
    col = "Adj Close" if "Adj Close" in data.columns else "Close"
    adj_close = data[col]
    if isinstance(adj_close, pd.Series):
        adj_close = adj_close.to_frame()
    return adj_close.dropna(how="all")

def rebalance_portfolio(weights, portfolio_value, current_prices):
    weights = np.array(weights)
    allocations = portfolio_value * weights
    return {t: alloc / current_prices[t] for t, alloc in zip(current_prices.index, allocations)}

def portfolio_value_from_shares(shares, current_prices):
    return sum(num * current_prices[t] for t, num in shares.items())

def backtest_portfolio(
    tickers, weights, start_year, end_year, initial_amount,
    monthly_contribution, rebalancing, benchmark_ticker
):
    start_date, end_date = f"{start_year}-01-01", f"{end_year}-12-31"
    all_tickers = list(tickers) + [benchmark_ticker]
    data = download_daily_data(all_tickers, start_date, end_date)
    benchmark_prices = data[benchmark_ticker]; prices = data.drop(columns=[benchmark_ticker])

    pv, bv = initial_amount, initial_amount
    shares = None
    p_vals, b_vals = [], []
    prev_m, prev_y = None, None

    def need_rebal(date):
        return (
            (prev_y is None) or
            (rebalancing == "annual" and date.year != prev_y) or
            (rebalancing == "monthly" and date.month != prev_m)
        )

    for i, d in enumerate(prices.index):
        if i == 0 or need_rebal(d):
            shares = rebalance_portfolio(weights, pv, prices.loc[d])
        pv = portfolio_value_from_shares(shares, prices.loc[d])

        if i and d.month != prev_m:              # aporte mensual
            pv += monthly_contribution
            if rebalancing != "none":
                shares = rebalance_portfolio(weights, pv, prices.loc[d])

        # benchmark
        if i:
            br = (benchmark_prices.loc[d] / benchmark_prices.iloc[i-1]) - 1
            bv *= 1 + br
            if d.month != prev_m:
                bv += monthly_contribution

        p_vals.append(pv); b_vals.append(bv)
        prev_m, prev_y = d.month, d.year

    return pd.DataFrame({"Portfolio": p_vals, "Benchmark": b_vals}, index=prices.index)

def calculate_performance_metrics(series, rf=0.0):
    initial, final = series.iloc[0], series.iloc[-1]
    total_ret = final / initial - 1
    years = (series.index[-1] - series.index[0]).days / 365.25
    cagr = (1 + total_ret) ** (1/years) - 1 if years else np.nan
    daily = series.pct_change().dropna()
    drawdown = (series / series.cummax()) - 1
    max_dd = drawdown.min()
    excess = daily - (((1+rf)**(1/252))-1)
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() else np.nan
    return {
        "CAGR": f"{cagr:.2%}",
        "Total Return": f"{total_ret:.2%}",
        "Max Drawdown": f"{max_dd:.2%}",
        "Sharpe": f"{sharpe:.2f}"
    }

# ------------------------- UI ----------------------------------
st.title("📈 Back-testing de Carteras")
col1, col2 = st.columns([2, 1])

with col1:
    tickers_str = st.text_input("Tickers (separados por coma)", "SPY,QQQ,TLT")
    weights_str = st.text_input("Pesos (mismos elementos, separados por coma)", "0.4,0.4,0.2")
    benchmark = st.text_input("Benchmark (ticker)", "^GSPC")
    st.caption("Los pesos deben sumar 1.0")

with col2:
    start_year = st.number_input("Año inicio", 1980, 2025, 2010, step=1)
    end_year   = st.number_input("Año fin", start_year, 2025, 2024, step=1)
    initial_amount = st.number_input("Monto inicial (USD)", 0, 10_000_000, 10000, step=1000)
    monthly_contribution = st.number_input("Aporte mensual", 0, 1_000_000, 0, step=100)
    rebalancing = st.selectbox("Rebalanceo", options=["none", "annual", "monthly"], index=1)

run = st.button("🎬 Ejecutar Back-test")

if run:
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    weights = [float(w) for w in weights_str.split(",")]
    if len(tickers) != len(weights):
        st.error("El número de tickers y de pesos debe coincidir.")
        st.stop()
    if abs(sum(weights) - 1) > 1e-6:
        st.error("La suma de los pesos debe ser 1.0")
        st.stop()

    st.info("Descargando datos y calculando…")
    results = backtest_portfolio(
        tickers, weights, start_year, end_year,
        initial_amount, monthly_contribution, rebalancing, benchmark
    )

    # ---------- Métricas y gráfico -----------------
    m = calculate_performance_metrics(results["Portfolio"])
    colA, colB, colC, colD = st.columns(4)
    colA.metric("CAGR", m["CAGR"])
    colB.metric("Return", m["Total Return"])
    colC.metric("Max DD", m["Max Drawdown"])
    colD.metric("Sharpe", m["Sharpe"])

    fig = px.line(results, title="Evolución del valor")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Datos diarios"):
        st.dataframe(results.tail(30))
