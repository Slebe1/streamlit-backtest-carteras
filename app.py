# app.py — versión multi-cartera para Streamlit
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Comparador de Carteras", layout="wide")

# ──────────── Funciones de negocio ────────────
@st.cache_data(show_spinner=False)
def download_daily_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date,
                       interval="1d", progress=False)
    if data.empty:
        raise ValueError(
            "No se pudieron descargar precios de Yahoo Finance. "
            "Ejecuta la app en un entorno con salida a Internet "
            "o usa datos locales.")
    col = "Adj Close" if "Adj Close" in data.columns else "Close"
    adj = data[col]
    if isinstance(adj, pd.Series):
        adj = adj.to_frame()
    return adj.dropna(how="all")

def rebalance_portfolio(weights, value, prices):
    w = np.array(weights)
    alloc = value * w
    return {t: a / prices[t] for t, a in zip(prices.index, alloc)}

def portfolio_value_from_shares(shares, prices):
    return sum(n * prices[t] for t, n in shares.items())

def backtest_portfolio(tickers, weights, start_year, end_year,
                       initial_amount, monthly, reb, bench):
    start, end = f"{start_year}-01-01", f"{end_year}-12-31"
    data = download_daily_data(list(tickers)+[bench], start, end)
    bench_px, prices = data[bench], data.drop(columns=[bench])

    pv, bv, shares = initial_amount, initial_amount, None
    p_vals, b_vals, pm, py = [], [], None, None

    for i, d in enumerate(prices.index):
        if i == 0 or (reb == "annual" and d.year != py) or (reb == "monthly" and d.month != pm):
            shares = rebalance_portfolio(weights, pv, prices.loc[d])

        pv = portfolio_value_from_shares(shares, prices.loc[d])

        if i and d.month != pm:
            pv += monthly
            if reb != "none":
                shares = rebalance_portfolio(weights, pv, prices.loc[d])

        if i:
            br = bench_px.loc[d] / bench_px.iloc[i-1] - 1
            bv = bv * (1 + br) + (monthly if d.month != pm else 0)

        p_vals.append(pv); b_vals.append(bv)
        pm, py = d.month, d.year

    return pd.DataFrame({"Portfolio": p_vals, "Benchmark": b_vals}, index=prices.index)

def perf(series, rf=0.0):
    if series.empty:
        return {"CAGR":"–", "Return":"–", "DD":"–", "Sharpe":"–"}
    init, fin = series.iloc[0], series.iloc[-1]
    tot = fin / init - 1
    yrs = (series.index[-1] - series.index[0]).days / 365.25
    cagr = (1 + tot) ** (1/yrs) - 1 if yrs else np.nan
    dd = (series / series.cummax() - 1).min()
    daily = series.pct_change().dropna()
    ex = daily - (((1 + rf) ** (1/252)) - 1)
    sharpe = np.sqrt(252) * ex.mean() / ex.std() if ex.std() else np.nan
    return {"CAGR": f"{cagr:.2%}", "Return": f"{tot:.2%}",
            "DD": f"{dd:.2%}", "Sharpe": f"{sharpe:.2f}"}

# ──────────── Interfaz ────────────
st.title("📈 Back-testing de Múltiples Carteras")

# Parámetros globales
gl_c1, gl_c2 = st.columns([2,1])
with gl_c1:
    n_port = st.number_input("Número de carteras a comparar", 1, 6, 2, step=1)
    bench = st.text_input("Ticker Benchmark", "^GSPC")
with gl_c2:
    y0 = st.number_input("Año inicio", 1980, 2025, 2010)
    y1 = st.number_input("Año fin", y0, 2025, 2024)
    rf = st.number_input("Tasa libre de riesgo anual (%)", 0.0, 20.0, 0.0) / 100

# Controles por cartera
portfolios_cfg = {}
for idx in range(int(n_port)):
    with st.expander(f"Cartera {idx+1}", expanded=(idx==0)):
        tk = st.text_input("Tickers (coma)", value="SPY,QQQ,TLT", key=f"tk{idx}")
        wt = st.text_input("Pesos (coma)", value="0.4,0.4,0.2", key=f"wt{idx}")
        init = st.number_input("Monto inicial", 0, 10_000_000, 10_000,
                               step=1000, key=f"init{idx}")
        mon = st.number_input("Aporte mensual", 0, 1_000_000, 0,
                              step=100, key=f"mon{idx}")
        reb = st.selectbox("Rebalanceo", ["none", "annual", "monthly"],
                           index=1, key=f"reb{idx}")
        portfolios_cfg[f"Cartera_{idx+1}"] = {
            "tickers": [t.strip().upper() for t in tk.split(",") if t.strip()],
            "weights": [float(w) for w in wt.split(",") if w.strip()],
            "initial_amount": init,
            "monthly": mon,
            "reb": reb
        }

# Botón principal
if st.button("🎬 Ejecutar Back-test"):
    # Validaciones rápidas
    for name, cfg in portfolios_cfg.items():
        if len(cfg["tickers"]) != len(cfg["weights"]):
            st.error(f"{name}: número de tickers ≠ número de pesos.")
            st.stop()
        if abs(sum(cfg["weights"]) - 1.0) > 1e-6:
            st.error(f"{name}: los pesos no suman 1.0.")
            st.stop()

    # Back-tests
    combined = pd.DataFrame()
    metrics = {}
    try:
        for name, cfg in portfolios_cfg.items():
            res = backtest_portfolio(
                cfg["tickers"], cfg["weights"],
                y0, y1, cfg["initial_amount"],
                cfg["monthly"], cfg["reb"], bench
            )
            combined[name] = res["Portfolio"]
            if "Benchmark" not in combined.columns:
                combined["Benchmark"] = res["Benchmark"]
            metrics[name] = perf(res["Portfolio"], rf)
    except ValueError as e:
        st.error(str(e)); st.stop()

    if combined.empty:
        st.error("Sin datos que mostrar."); st.stop()

    # Métricas resumen
    st.subheader("Resumen")
    st.write(
        pd.DataFrame(metrics).T.rename(columns={
            "CAGR":"CAGR", "Return":"Retorno", "DD":"Máx DD", "Sharpe":"Sharpe"
        })
    )

    # Gráfico evolutivo
    st.subheader("Evolución")
    st.plotly_chart(px.line(combined, title="Valor de las carteras"),
                    use_container_width=True)

    # Tabla datos crudos
    with st.expander("Datos diarios (últimas filas)"):
        st.dataframe(combined.tail(30))
