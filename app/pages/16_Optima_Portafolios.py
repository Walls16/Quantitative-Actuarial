"""
app.py — Optima
================
Plataforma profesional de optimizacion de portafolios.
Inspirada en Bloomberg / FactSet. Tema oscuro, datos en tiempo real desde Yahoo Finance.
"""

import datetime as dt
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

from app.optima_services import download_prices, get_live_quotes
from quantitativeactuarial.portfolioopt import (
    PortfolioOptimizer,
    SCENARIOS,
    STRATEGY_NAMES,
    compute_all_metrics,
    cumulative_returns,
    get_returns,
    markowitz_frontier,
    monte_carlo_portfolios,
    portfolio_returns as port_ret_fn,
    run_all_scenarios,
    run_backtest,
    validate_tickers,
)
from app.optima_charts import (
    plot_price_history,
    plot_efficient_frontier,
    plot_equity_curve,
    plot_drawdown,
    plot_weights_over_time,
    plot_stress_scenario,
    plot_stress_summary,
    plot_strategy_comparison,
    plot_correlation_heatmap,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Optima | Portfolio Optimizer",
    page_icon="O",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0e1a !important;
    color: #e0e6f0 !important;
}
[data-testid="stSidebar"] {
    background-color: #0d1220 !important;
    border-right: 1px solid #1e2d4a;
}
.logo-header {
    font-family: 'Courier New', monospace;
    font-size: 30px;
    font-weight: 900;
    color: #00D4FF;
    letter-spacing: 4px;
    text-transform: uppercase;
    border-bottom: 2px solid #00D4FF;
    padding-bottom: 8px;
    margin-bottom: 4px;
}
.logo-sub {
    font-size: 11px;
    color: #5a7a9a;
    letter-spacing: 2px;
    font-family: 'Courier New', monospace;
    margin-bottom: 16px;
}
.ticker-bar {
    background: #0d1220;
    border: 1px solid #1e2d4a;
    border-radius: 6px;
    padding: 10px 16px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    display: flex;
    gap: 28px;
    flex-wrap: wrap;
    align-items: center;
    margin-bottom: 14px;
}
.ticker-item { display: inline-flex; gap: 6px; align-items: center; }
.ticker-sym   { color: #00D4FF; font-weight: 700; }
.ticker-price { color: #e0e6f0; }
.ticker-up    { color: #00FA9A; }
.ticker-down  { color: #FF4B4B; }
.metric-card {
    background: #0f1626;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 14px 18px;
    text-align: center;
    min-width: 110px;
}
.metric-label { font-size: 11px; color: #5a7a9a; font-family: 'Courier New', monospace; letter-spacing: 1px; }
.metric-value { font-size: 21px; font-weight: 700; color: #00D4FF; font-family: 'Courier New', monospace; }
.metric-value.pos { color: #00FA9A; }
.metric-value.neg { color: #FF4B4B; }
.section-title {
    font-family: 'Courier New', monospace;
    font-size: 12px;
    color: #5a7a9a;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-bottom: 1px solid #1e2d4a;
    padding-bottom: 4px;
    margin: 18px 0 12px 0;
}
.stress-card {
    background: #0f1626;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.stress-title { color: #FFD700; font-weight: 700; font-size: 14px; margin-bottom: 4px; }
.stress-desc  { color: #7a9ab8; font-size: 12px; margin-bottom: 8px; }
.control-panel {
    background: #0d1220;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 16px 18px 10px 18px;
    margin: 12px 0 16px 0;
}
.control-note {
    color: #7a9ab8;
    font-size: 12px;
    margin-top: -4px;
    margin-bottom: 8px;
}
[data-testid="stTab"] { font-size: 13px; }
.stButton > button {
    background: #00D4FF18;
    color: #00D4FF;
    border: 1px solid #00D4FF55;
    border-radius: 5px;
    font-family: 'Courier New', monospace;
    font-weight: 700;
    letter-spacing: 1px;
    transition: all 0.2s;
}
.stButton > button:hover { background: #00D4FF33; border-color: #00D4FF; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)

# Auto-refresh cada 60 s para el ticker bar
st.markdown('<meta http-equiv="refresh" content="60">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACION EN PAGINA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="logo-header">OPTIMA</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="logo-sub">PROFESSIONAL PORTFOLIO OPTIMIZER v2.0</div>', unsafe_allow_html=True
)
st.markdown('<div class="control-panel">', unsafe_allow_html=True)

cfg_left, cfg_mid, cfg_right = st.columns([1.25, 1, 1], gap="large")

with cfg_left:
    st.markdown('<div class="section-title">Universo de Activos</div>', unsafe_allow_html=True)
    tickers_input = st.text_area(
        "Tickers (uno por linea o separados por coma)",
        value="AAPL\nMSFT\nGOOGL\nAMZN\nNVDA\nBRK-B\nJNJ\nGLD\nTLT",
        height=176,
        help="Acciones, ETFs, bonos, commodities, cripto — cualquier activo de Yahoo Finance",
    )
    raw_tickers = [
        t.strip().upper() for t in tickers_input.replace(",", "\n").splitlines() if t.strip()
    ]
    tickers = validate_tickers(raw_tickers)

with cfg_mid:
    st.markdown('<div class="section-title">Periodo Historico</div>', unsafe_allow_html=True)
    date_start_col, date_end_col = st.columns(2)
    with date_start_col:
        start_date = st.date_input("Inicio", value=dt.date(2019, 1, 1), max_value=dt.date.today())
    with date_end_col:
        end_date = st.date_input("Fin", value=dt.date.today(), max_value=dt.date.today())

    st.markdown('<div class="section-title">Optimizacion</div>', unsafe_allow_html=True)
    selected_strategy = st.selectbox("Estrategia Principal", STRATEGY_NAMES, index=1)
    rf_rate = st.slider("Tasa Libre de Riesgo (%)", 0.0, 10.0, 4.5, 0.1) / 100
    allow_short = st.toggle("Permitir posiciones en corto", value=False)

with cfg_right:
    st.markdown('<div class="section-title">Backtesting</div>', unsafe_allow_html=True)
    rebal_freq = st.selectbox(
        "Frecuencia de Rebalanceo", ["Mensual", "Trimestral", "Anual"], index=0
    )
    lookback = st.slider(
        "Ventana lookback (dias)",
        63,
        504,
        252,
        21,
        help="Dias de historia usados para estimar parametros en cada rebalanceo",
    )
    initial_capital = st.number_input(
        "Capital Inicial ($)", min_value=1000, value=100_000, step=1000
    )

    st.markdown('<div class="section-title">Watchlist</div>', unsafe_allow_html=True)
    watchlist_input = st.text_input(
        "Tickers watchlist",
        value="SPY,QQQ,BTC-USD,GLD,TLT,AAPL,NVDA,^VIX",
        help="Separados por coma",
    )
    watchlist = tuple(t.strip().upper() for t in watchlist_input.split(",") if t.strip())

run_col, hint_col = st.columns([0.28, 0.72])
with run_col:
    run_btn = st.button("EJECUTAR ANALISIS", use_container_width=True)
with hint_col:
    st.markdown(
        '<div class="control-note">Configura el universo, periodo y parametros; luego ejecuta el analisis para actualizar resultados.</div>',
        unsafe_allow_html=True,
    )

st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TICKER BAR
# ─────────────────────────────────────────────────────────────────────────────
def render_ticker_bar(tks):
    quotes = get_live_quotes(tks)
    if quotes.empty:
        return
    items = ""
    for _, row in quotes.iterrows():
        sign = "+" if row["Cambio %"] >= 0 else ""
        cls = "ticker-up" if row["Cambio %"] >= 0 else "ticker-down"
        arr = "▲" if row["Cambio %"] >= 0 else "▼"
        items += (
            f'<span class="ticker-item">'
            f'<span class="ticker-sym">{row["Ticker"]}</span>'
            f'<span class="ticker-price">${row["Precio"]:,.2f}</span>'
            f'<span class="{cls}">{arr} {sign}{row["Cambio %"]:.2f}%</span>'
            f"</span>"
        )
    st.markdown(f'<div class="ticker-bar">{items}</div>', unsafe_allow_html=True)


render_ticker_bar(watchlist)

# ─────────────────────────────────────────────────────────────────────────────
# ESTADO DE SESION
# ─────────────────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ─────────────────────────────────────────────────────────────────────────────
# PANTALLA DE BIENVENIDA
# ─────────────────────────────────────────────────────────────────────────────
if not run_btn and st.session_state.results is None:
    st.markdown("## Optima — Portfolio Optimizer")
    st.markdown("""
> Configura tu universo de activos en el panel superior y presiona **EJECUTAR ANALISIS**.
---
""")
    c1, c2, c3 = st.columns(3)
    c1.info(
        "**19 Estrategias**\nMax Sharpe · Min Varianza · ERC · CVaR · HRP · Kelly · Black-Litterman y mas"
    )
    c2.info(
        "**Backtesting dinamico**\nRebalanceo mensual, trimestral o anual con ventana walk-forward real"
    )
    c3.info("**Stress Testing**\nCrisis 2008 · COVID-19 · Dot-com · Subida de tasas Fed 2022")
    d1, d2, d3 = st.columns(3)
    d1.success("**Frontera Eficiente**\nMarkowitz analitico + 3,000 portafolios Monte Carlo")
    d2.success(
        "**Datos en tiempo real**\nYahoo Finance · Acciones · ETFs · Bonos · Cripto · Commodities"
    )
    d3.success(
        "**Metricas completas**\nSharpe · Sortino · VaR · CVaR · Max Drawdown · Calmar · CDaR"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y PROCESAMIENTO DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    if len(tickers) < 2:
        st.error("Necesitas al menos 2 tickers para optimizar un portafolio.")
        st.stop()

    with st.spinner("Descargando datos de mercado desde Yahoo Finance..."):
        prices = download_prices(tuple(tickers), start=start_date, end=end_date)

    if prices.empty or prices.shape[1] < 2:
        st.error("No se pudieron obtener datos. Verifica los tickers y el periodo seleccionado.")
        st.stop()

    valid_cols = prices.columns[prices.notna().mean() >= 0.7].tolist()
    if len(valid_cols) < 2:
        st.error("No hay suficientes activos con datos historicos en el periodo seleccionado.")
        st.stop()
    prices = prices[valid_cols].dropna()
    tickers_valid = list(prices.columns)
    returns = get_returns(prices)

    with st.spinner("Optimizando portafolios con las 19 estrategias..."):
        optimizer = PortfolioOptimizer(returns, rf=rf_rate, allow_short=allow_short)
        primary_w = optimizer.optimize(selected_strategy)
        primary_weights = pd.Series(primary_w, index=tickers_valid)

        all_weights, all_metrics = {}, {}
        for strat in STRATEGY_NAMES:
            try:
                w = optimizer.optimize(strat)
                ws = pd.Series(w, index=tickers_valid)
                pr = port_ret_fn(returns, w)
                m = compute_all_metrics(pr, rf=rf_rate)
                all_weights[strat] = ws
                all_metrics[strat] = m
            except Exception:
                continue

    with st.spinner("Calculando frontera eficiente y Monte Carlo..."):
        frontier_df = markowitz_frontier(
            optimizer.mean_returns, optimizer.cov_matrix, n_points=60, allow_short=allow_short
        )
        mc_df = monte_carlo_portfolios(
            optimizer.mean_returns,
            optimizer.cov_matrix,
            n_portfolios=3000,
            rf=rf_rate,
            allow_short=allow_short,
        )

    with st.spinner("Ejecutando backtesting walk-forward..."):
        bt = run_backtest(
            prices,
            strategy_name=selected_strategy,
            rebalance_freq=rebal_freq,
            lookback_days=lookback,
            rf=rf_rate,
            allow_short=allow_short,
            initial_capital=float(initial_capital),
        )

    with st.spinner("Corriendo stress tests historicos..."):
        prices_long = download_prices(
            tuple(tickers_valid), start=dt.date(1999, 1, 1), end=dt.date.today()
        )
        if prices_long.empty:
            prices_long = prices
        stress_results = run_all_scenarios(prices_long, primary_weights)

    st.session_state.results = dict(
        prices=prices,
        returns=returns,
        tickers=tickers_valid,
        optimizer=optimizer,
        primary_weights=primary_weights,
        selected_strategy=selected_strategy,
        all_weights=all_weights,
        all_metrics=all_metrics,
        frontier_df=frontier_df,
        mc_df=mc_df,
        bt=bt,
        stress_results=stress_results,
        rf_rate=rf_rate,
    )

R = st.session_state.results
if R is None:
    st.stop()

prices = R["prices"]
returns = R["returns"]
tickers_valid = R["tickers"]
optimizer = R["optimizer"]
primary_weights = R["primary_weights"]
selected_strategy = R["selected_strategy"]
all_weights = R["all_weights"]
all_metrics = R["all_metrics"]
frontier_df = R["frontier_df"]
mc_df = R["mc_df"]
bt = R["bt"]
stress_results = R["stress_results"]
rf_rate = R["rf_rate"]

# ─────────────────────────────────────────────────────────────────────────────
# KPI HEADER
# ─────────────────────────────────────────────────────────────────────────────
primary_port_ret = port_ret_fn(returns, primary_weights.values)
primary_metrics = compute_all_metrics(primary_port_ret, rf=rf_rate)

kpi_data = [
    ("RETORNO ANUAL.", primary_metrics["Retorno Anualizado"], "pct", True),
    ("VOLATILIDAD", primary_metrics["Volatilidad Anualizada"], "pct_abs", False),
    ("SHARPE", primary_metrics["Sharpe Ratio"], "float", True),
    ("SORTINO", primary_metrics["Sortino Ratio"], "float", True),
    ("MAX DRAWDOWN", primary_metrics["Max Drawdown"], "pct", True),
    ("CALMAR", primary_metrics["Calmar Ratio"], "float", True),
    ("VaR 95% (d)", primary_metrics["VaR 95% (diario)"], "pct", True),
    ("CVaR 95% (d)", primary_metrics["CVaR 95% (diario)"], "pct", True),
]

kpi_html = '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px;">'
for label, val, fmt, good_pos in kpi_data:
    if fmt == "pct":
        display = f"{val * 100:+.2f}%"
    elif fmt == "pct_abs":
        display = f"{abs(val) * 100:.2f}%"
    else:
        display = f"{val:.2f}"
    cls = ("pos" if val >= 0 else "neg") if good_pos else ("neg" if val >= 0 else "pos")
    kpi_html += (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value {cls}">{display}</div>'
        f"</div>"
    )
kpi_html += "</div>"

st.markdown(
    f"### Estrategia activa: "
    f"<span style=\"font-family:'Courier New',monospace;color:#FFD700;font-size:14px;font-weight:700;\">"
    f"{selected_strategy}</span>",
    unsafe_allow_html=True,
)
st.markdown(kpi_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: grafico de barras horizontales para pesos
# ─────────────────────────────────────────────────────────────────────────────
def plot_weights_horizontal(weights: pd.Series, title: str = "Pesos del Portafolio") -> go.Figure:
    w = weights[weights.abs() > 1e-4].sort_values(ascending=True)
    colors = ["#FF4B4B" if v < 0 else "#00D4FF" for v in w.values]
    fig = go.Figure(
        data=[
            go.Bar(
                x=w.values * 100,
                y=w.index,
                orientation="h",
                marker_color=colors,
                text=[f"{v * 100:.1f}%" for v in w.values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Peso (%)",
        margin=dict(l=10, r=60, t=50, b=10),
        height=max(300, 40 * len(w)),
        plot_bgcolor="#0a0e1a",
        paper_bgcolor="#0a0e1a",
    )
    return fig


def plot_weights_vertical(weights: pd.Series, title: str = "Distribucion de Pesos") -> go.Figure:
    w = weights[weights.abs() > 1e-4].sort_values(ascending=False)
    colors = ["#FF4B4B" if v < 0 else "#00D4FF" for v in w.values]
    fig = go.Figure(
        data=[
            go.Bar(
                x=w.index,
                y=w.values * 100,
                marker_color=colors,
                text=[f"{v * 100:.1f}%" for v in w.values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        title=title,
        yaxis_title="Peso (%)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=380,
        plot_bgcolor="#0a0e1a",
        paper_bgcolor="#0a0e1a",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "Optimizacion",
        "Backtesting",
        "Stress Testing",
        "Frontera Eficiente",
        "Comparar Estrategias",
        "Datos y Correlaciones",
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 · OPTIMIZACION
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        f'<div class="section-title">Asignacion Optima — {selected_strategy}</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_weights_horizontal(primary_weights, "Pesos — Barras Horizontales"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with c2:
        st.plotly_chart(
            plot_weights_vertical(primary_weights, "Pesos — Barras Verticales"),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown('<div class="section-title">Pesos Detallados</div>', unsafe_allow_html=True)
    w_df = pd.DataFrame(
        {
            "Ticker": primary_weights.index,
            "Peso (%)": (primary_weights * 100).round(2),
        }
    ).reset_index(drop=True)
    st.dataframe(
        w_df, use_container_width=True, hide_index=True, height=min(420, 45 * (len(w_df) + 1))
    )

    st.markdown(
        '<div class="section-title">Metricas de Riesgo / Rendimiento</div>', unsafe_allow_html=True
    )
    metrics_df = pd.DataFrame.from_dict(primary_metrics, orient="index", columns=["Valor"])
    metrics_df["Valor"] = metrics_df["Valor"].apply(lambda x: f"{x:.4f}")
    st.dataframe(metrics_df, use_container_width=True)

    st.markdown(
        '<div class="section-title">Retorno Acumulado del Portafolio</div>', unsafe_allow_html=True
    )
    cum = cumulative_returns(primary_port_ret)
    eq_cum = cumulative_returns(returns.mean(axis=1))
    fig_cum = go.Figure()
    fig_cum.add_trace(
        go.Scatter(
            x=cum.index,
            y=(cum - 1) * 100,
            mode="lines",
            fill="tozeroy",
            name=selected_strategy,
            line=dict(color="#00D4FF", width=2),
            fillcolor="rgba(0,212,255,0.06)",
        )
    )
    fig_cum.add_trace(
        go.Scatter(
            x=eq_cum.index,
            y=(eq_cum - 1) * 100,
            mode="lines",
            name="Equal Weight",
            line=dict(color="#FFA500", width=1.5, dash="dash"),
        )
    )
    fig_cum.add_hline(y=0, line_dash="dot", line_color="#444")
    fig_cum.update_layout(
        template="plotly_dark",
        title="Retorno Acumulado (hold desde inicio del periodo)",
        xaxis_title="Fecha",
        yaxis_title="Retorno acumulado (%)",
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
        plot_bgcolor="#0a0e1a",
        paper_bgcolor="#0a0e1a",
    )
    st.plotly_chart(fig_cum, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 · BACKTESTING
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    if bt is None:
        st.warning(
            f"No hay suficientes datos para el backtesting con ventana de {lookback} dias "
            f"y frecuencia {rebal_freq}. Prueba un periodo historico mas largo o ventana mas corta."
        )
    else:
        bt_m = compute_all_metrics(bt["returns"], rf=rf_rate)
        bch_m = compute_all_metrics(bt["benchmark_returns"], rf=rf_rate)

        st.markdown(
            f'<div class="section-title">Resultado del Backtest — {selected_strategy} | {rebal_freq}</div>',
            unsafe_allow_html=True,
        )

        cbt1, cbt2, cbt3, cbt4 = st.columns(4)
        cbt1.metric(
            "Retorno Anual.",
            f"{bt_m['Retorno Anualizado'] * 100:+.2f}%",
            f"vs EW: {(bt_m['Retorno Anualizado'] - bch_m['Retorno Anualizado']) * 100:+.2f}%",
        )
        cbt2.metric(
            "Sharpe",
            f"{bt_m['Sharpe Ratio']:.3f}",
            f"vs EW: {bt_m['Sharpe Ratio'] - bch_m['Sharpe Ratio']:+.3f}",
        )
        cbt3.metric(
            "Max Drawdown",
            f"{bt_m['Max Drawdown'] * 100:.2f}%",
            f"vs EW: {(bt_m['Max Drawdown'] - bch_m['Max Drawdown']) * 100:+.2f}%",
        )
        cbt4.metric("Rebalanceos", len(bt["rebalance_dates"]))

        st.plotly_chart(
            plot_equity_curve(bt["equity_curve"], bt["benchmark_equity"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.plotly_chart(
            plot_drawdown(bt["returns"], bt["benchmark_returns"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        if not bt["weights_history"].empty:
            st.plotly_chart(
                plot_weights_over_time(bt["weights_history"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        st.markdown(
            '<div class="section-title">Metricas: Portafolio vs Benchmark (Equal Weight)</div>',
            unsafe_allow_html=True,
        )
        comp_rows = {
            k: {
                selected_strategy: f"{bt_m[k]:.4f}",
                "Equal Weight (BH)": f"{bch_m.get(k, float('nan')):.4f}",
            }
            for k in bt_m
        }
        st.dataframe(pd.DataFrame(comp_rows).T, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 · STRESS TESTING
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(
        '<div class="section-title">Analisis de Crisis Historicas</div>', unsafe_allow_html=True
    )

    if not any(v is not None for v in stress_results.values()):
        st.warning(
            "No hay datos historicos suficientes para ningun escenario con los activos seleccionados."
        )
    else:
        st.plotly_chart(
            plot_stress_summary(stress_results),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        for scenario_name, result in stress_results.items():
            info = SCENARIOS[scenario_name]
            with st.expander(f"{scenario_name}", expanded=False):
                if result is None:
                    st.info(
                        "No hay suficientes datos historicos para este escenario con los activos seleccionados."
                    )
                    continue

                st.markdown(
                    f'<div class="stress-card">'
                    f'<div class="stress-title">{scenario_name}</div>'
                    f'<div class="stress-desc">{info["descripcion"]}</div>'
                    f'<div style="font-size:12px;color:#5a7a9a;">'
                    f"Periodo: {result['period_start'].date()} — {result['period_end'].date()}"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
                if result["excluded_assets"]:
                    st.warning(
                        f"Activos excluidos por falta de datos: {', '.join(result['excluded_assets'])}. "
                        "Los pesos se renormalizaron."
                    )

                cs1, cs2, cs3, cs4 = st.columns(4)
                cs1.metric(
                    "Retorno Total",
                    f"{result['total_return'] * 100:.2f}%",
                    f"vs EW: {(result['total_return'] - result['benchmark_total_return']) * 100:+.2f}%",
                )
                cs2.metric(
                    "Max Drawdown",
                    f"{result['max_drawdown'] * 100:.2f}%",
                    f"vs EW: {(result['max_drawdown'] - result['benchmark_max_drawdown']) * 100:+.2f}%",
                )
                cs3.metric("Volatilidad Anual.", f"{result['volatility'] * 100:.2f}%")
                cs4.metric("CVaR 95% (diario)", f"{result['cvar_95'] * 100:.2f}%")

                st.plotly_chart(
                    plot_stress_scenario(result),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

                df_w = (result["weights_used"] * 100).round(2).to_frame("Peso (%)")
                st.dataframe(df_w, use_container_width=True, height=min(300, 40 * (len(df_w) + 1)))

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 · FRONTERA EFICIENTE
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(
        '<div class="section-title">Frontera Eficiente de Markowitz + Nube Monte Carlo (3,000 portafolios)</div>',
        unsafe_allow_html=True,
    )

    special_pts = {}
    for strat in ["Maximo Sharpe", "Minima Varianza", selected_strategy]:
        if strat in all_weights:
            w = all_weights[strat].values
            special_pts[strat] = (
                float(np.sqrt(w @ optimizer.cov_matrix.values @ w)),
                float(w @ optimizer.mean_returns.values),
            )

    st.plotly_chart(
        plot_efficient_frontier(mc_df, frontier_df, special_pts),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown(
        '<div class="section-title">Top 10 Portafolios Monte Carlo (Mayor Sharpe)</div>',
        unsafe_allow_html=True,
    )
    top_mc = mc_df.nlargest(10, "Sharpe")[
        ["Retorno", "Volatilidad", "Sharpe"] + tickers_valid
    ].copy()
    top_mc["Retorno"] = (top_mc["Retorno"] * 100).round(2)
    top_mc["Volatilidad"] = (top_mc["Volatilidad"] * 100).round(2)
    top_mc["Sharpe"] = top_mc["Sharpe"].round(3)
    for t in tickers_valid:
        top_mc[t] = (top_mc[t] * 100).round(1)
    top_mc.columns = ["Ret (%)", "Vol (%)", "Sharpe"] + [f"{t} (%)" for t in tickers_valid]
    st.dataframe(top_mc.reset_index(drop=True), use_container_width=True, hide_index=True)

    if not frontier_df.empty:
        min_vol_row = frontier_df.loc[frontier_df["Volatilidad"].idxmin()]
        max_sh_row = frontier_df.loc[
            (frontier_df["Retorno"] / frontier_df["Volatilidad"].replace(0, np.nan)).idxmax()
        ]
        cf1, cf2 = st.columns(2)
        cf1.info(
            f"**Portafolio de Minima Varianza** (Frontera Analitica)\n\n"
            f"Volatilidad: {min_vol_row['Volatilidad'] * 100:.2f}%  |  Retorno: {min_vol_row['Retorno'] * 100:.2f}%"
        )
        cf2.success(
            f"**Portafolio de Maximo Sharpe** (Frontera Analitica)\n\n"
            f"Volatilidad: {max_sh_row['Volatilidad'] * 100:.2f}%  |  Retorno: {max_sh_row['Retorno'] * 100:.2f}%"
            f"  |  Sharpe: {max_sh_row['Sharpe']:.3f}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 · COMPARAR ESTRATEGIAS
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown(
        f'<div class="section-title">Comparacion de {len(all_metrics)} Estrategias de Optimizacion</div>',
        unsafe_allow_html=True,
    )

    if not all_metrics:
        st.warning("No se pudo calcular las metricas de todas las estrategias.")
    else:
        metrics_compare = pd.DataFrame(all_metrics).T
        metrics_compare.index.name = "Estrategia"

        # ─────────────────────────────────────────────────────────────────
        # SELECTOR: "Mejor estrategia segun tu criterio"
        # ─────────────────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-title">Mejor Estrategia Segun tu Criterio</div>',
            unsafe_allow_html=True,
        )

        BEST_CRITERIA = {
            "Mayor Sharpe Ratio": ("Sharpe Ratio", False),
            "Mayor Retorno Anualizado": ("Retorno Anualizado", False),
            "Menor Volatilidad": ("Volatilidad Anualizada", True),
            "Mayor Sortino Ratio": ("Sortino Ratio", False),
            "Mayor Calmar Ratio": ("Calmar Ratio", False),
            "Menor Max Drawdown": ("Max Drawdown", True),
            "Menor CVaR 95% (diario)": ("CVaR 95% (diario)", True),
        }
        crit_cols = st.columns([2, 3])
        with crit_cols[0]:
            criterio_label = st.selectbox(
                "Elegir la mejor por:", list(BEST_CRITERIA.keys()), index=0
            )

        col_name, lower_is_better = BEST_CRITERIA[criterio_label]
        if col_name in metrics_compare.columns:
            col_vals = metrics_compare[col_name]
            best_strategy_name = col_vals.idxmin() if lower_is_better else col_vals.idxmax()
            best_value = col_vals[best_strategy_name]

            with crit_cols[1]:
                val_str = (
                    f"{best_value * 100:.2f}%"
                    if any(k in col_name for k in ["Retorno", "Volatilidad", "Drawdown", "VaR"])
                    else f"{best_value:.3f}"
                )
                st.success(f"**Ganadora: {best_strategy_name}**  —  {col_name}: {val_str}")

            # Grafico comparativo: la ganadora resaltada vs el resto de estrategias
            comp_metric_cols = [
                "Sharpe Ratio",
                "Retorno Anualizado",
                "Volatilidad Anualizada",
                "Max Drawdown",
            ]
            comp_metric_cols = [c for c in comp_metric_cols if c in metrics_compare.columns]

            fig_best = go.Figure()
            x_strats = metrics_compare.index.tolist()
            bar_colors = ["#FFD700" if s == best_strategy_name else "#1e3a5a" for s in x_strats]
            line_colors = ["#FFD700" if s == best_strategy_name else "#3a5a7a" for s in x_strats]

            sharpe_vals = (
                metrics_compare["Sharpe Ratio"]
                if "Sharpe Ratio" in metrics_compare.columns
                else None
            )
            if sharpe_vals is not None:
                fig_best.add_trace(
                    go.Bar(
                        x=x_strats,
                        y=sharpe_vals.values,
                        marker_color=bar_colors,
                        marker_line_color=line_colors,
                        marker_line_width=1.5,
                        name="Sharpe Ratio",
                        text=[f"{v:.2f}" for v in sharpe_vals.values],
                        textposition="outside",
                    )
                )
            fig_best.update_layout(
                template="plotly_dark",
                title=f"Sharpe Ratio por Estrategia — Ganadora resaltada: {best_strategy_name}",
                yaxis_title="Sharpe Ratio",
                xaxis_tickangle=-35,
                height=420,
                margin=dict(l=10, r=10, t=55, b=120),
                showlegend=False,
                plot_bgcolor="#0a0e1a",
                paper_bgcolor="#0a0e1a",
            )
            st.plotly_chart(fig_best, use_container_width=True, config={"displayModeBar": False})

            # Curva de retorno acumulado: ganadora vs Equal Weight vs estrategia principal seleccionada
            if best_strategy_name in all_weights:
                best_w = all_weights[best_strategy_name].values
                best_ret = port_ret_fn(returns, best_w)
                best_cum = cumulative_returns(best_ret)
                ew_cum = cumulative_returns(returns.mean(axis=1))

                fig_cum_best = go.Figure()
                fig_cum_best.add_trace(
                    go.Scatter(
                        x=best_cum.index,
                        y=(best_cum - 1) * 100,
                        mode="lines",
                        name=f"Ganadora: {best_strategy_name}",
                        line=dict(color="#FFD700", width=2.5),
                    )
                )
                if selected_strategy != best_strategy_name and selected_strategy in all_weights:
                    sel_ret = port_ret_fn(returns, all_weights[selected_strategy].values)
                    sel_cum = cumulative_returns(sel_ret)
                    fig_cum_best.add_trace(
                        go.Scatter(
                            x=sel_cum.index,
                            y=(sel_cum - 1) * 100,
                            mode="lines",
                            name=f"Tu seleccion: {selected_strategy}",
                            line=dict(color="#00D4FF", width=1.8, dash="dot"),
                        )
                    )
                fig_cum_best.add_trace(
                    go.Scatter(
                        x=ew_cum.index,
                        y=(ew_cum - 1) * 100,
                        mode="lines",
                        name="Equal Weight",
                        line=dict(color="#7a9ab8", width=1.3, dash="dash"),
                    )
                )
                fig_cum_best.add_hline(y=0, line_dash="dot", line_color="#444")
                fig_cum_best.update_layout(
                    template="plotly_dark",
                    title="Retorno Acumulado — Ganadora vs Tu Seleccion vs Equal Weight",
                    xaxis_title="Fecha",
                    yaxis_title="Retorno acumulado (%)",
                    height=380,
                    margin=dict(l=10, r=10, t=50, b=10),
                    legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
                    plot_bgcolor="#0a0e1a",
                    paper_bgcolor="#0a0e1a",
                )
                st.plotly_chart(
                    fig_cum_best, use_container_width=True, config={"displayModeBar": False}
                )

                # Tabla resumen rapida: ganadora vs seleccion vs equal weight
                ew_m = compute_all_metrics(returns.mean(axis=1), rf=rf_rate)
                best_m = all_metrics[best_strategy_name]
                rows = {
                    f"Ganadora ({best_strategy_name})": best_m,
                    f"Tu seleccion ({selected_strategy})": all_metrics.get(selected_strategy, {}),
                    "Equal Weight": ew_m,
                }
                summary_df = pd.DataFrame(rows).T
                show_cols = [
                    c
                    for c in [
                        "Retorno Anualizado",
                        "Volatilidad Anualizada",
                        "Sharpe Ratio",
                        "Sortino Ratio",
                        "Max Drawdown",
                        "Calmar Ratio",
                    ]
                    if c in summary_df.columns
                ]
                summary_fmt = summary_df[show_cols].copy()
                for c in show_cols:
                    if any(k in c for k in ["Retorno", "Volatilidad", "Drawdown"]):
                        summary_fmt[c] = (summary_fmt[c] * 100).round(2).astype(str) + "%"
                    else:
                        summary_fmt[c] = summary_fmt[c].round(3)
                st.dataframe(summary_fmt, use_container_width=True)

        metric_options = list(metrics_compare.columns)
        selected_metric = st.selectbox(
            "Metrica para comparar (todas las estrategias):", metric_options, index=2
        )

        st.plotly_chart(
            plot_strategy_comparison(metrics_compare, selected_metric),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.markdown(
            '<div class="section-title">Tabla Comparativa Completa</div>', unsafe_allow_html=True
        )

        fmt_cols = {
            col: ("{:.3f}" if "Ratio" in col else "{:.4f}") for col in metrics_compare.columns
        }

        def highlight_best(s):
            is_lower_better = any(x in s.name for x in ["Drawdown", "VaR", "CVaR"])
            best = s.min() if is_lower_better else s.max()
            return ["background-color: rgba(0,250,154,0.15)" if v == best else "" for v in s]

        st.dataframe(
            metrics_compare.style.apply(highlight_best, axis=0).format(fmt_cols),
            use_container_width=True,
            height=560,
        )

        st.markdown(
            '<div class="section-title">Ranking Global (Puntuacion Compuesta)</div>',
            unsafe_allow_html=True,
        )
        rank_cols = [
            c
            for c in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Retorno Anualizado"]
            if c in metrics_compare.columns
        ]
        if rank_cols:
            score = pd.Series(0.0, index=metrics_compare.index)
            for c in rank_cols:
                col_s = metrics_compare[c]
                rng = col_s.max() - col_s.min()
                if rng > 0:
                    score += (col_s - col_s.min()) / rng
            score /= len(rank_cols)
            rank_df = score.sort_values(ascending=False).rename("Score (0-1)").to_frame()
            rank_df["Score (0-1)"] = rank_df["Score (0-1)"].round(4)
            rank_df.insert(0, "Posicion", range(1, len(rank_df) + 1))
            st.dataframe(rank_df, use_container_width=True)

        st.markdown(
            '<div class="section-title">Asignacion de Pesos por Estrategia (%)</div>',
            unsafe_allow_html=True,
        )
        weights_compare = (pd.DataFrame(all_weights).T * 100).round(2)
        st.dataframe(weights_compare, use_container_width=True, height=520)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 · DATOS Y CORRELACIONES
# ─────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown(
        '<div class="section-title">Evolucion de Precios Historicos</div>', unsafe_allow_html=True
    )
    st.plotly_chart(
        plot_price_history(prices),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown('<div class="section-title">Matriz de Correlacion</div>', unsafe_allow_html=True)
    st.plotly_chart(
        plot_correlation_heatmap(returns.corr()),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown(
        '<div class="section-title">Estadisticas Descriptivas de Retornos Diarios</div>',
        unsafe_allow_html=True,
    )
    desc = returns.describe().T.rename(columns={"mean": "Media", "std": "Desv.Est."})
    for c in desc.columns:
        try:
            desc[c] = (desc[c] * 100).round(4)
        except Exception:
            pass
    st.dataframe(desc, use_container_width=True)

    st.markdown(
        '<div class="section-title">Ultimos 30 dias de Precios de Cierre</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(prices.tail(30).round(2), use_container_width=True)

    st.markdown(
        '<div class="section-title">Pares con Alta Correlacion (|rho| > 0.7)</div>',
        unsafe_allow_html=True,
    )
    corr_m = returns.corr()
    n = len(tickers_valid)
    pairs = [
        {
            "Par": f"{tickers_valid[i]} / {tickers_valid[j]}",
            "Correlacion": round(corr_m.iloc[i, j], 4),
            "Tipo": "Alta positiva" if corr_m.iloc[i, j] > 0 else "Alta negativa",
        }
        for i in range(n)
        for j in range(i + 1, n)
        if abs(corr_m.iloc[i, j]) > 0.7
    ]
    if pairs:
        st.dataframe(pd.DataFrame(pairs), use_container_width=True, hide_index=True)
    else:
        st.info("Ningun par de activos supera |rho| = 0.7 — buena diversificacion.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style=\"text-align:center;font-size:11px;color:#3a5a7a;font-family:'Courier New',monospace;\">"
    "Optima v2.0 · Datos: Yahoo Finance · Optimizacion: SciPy SLSQP · "
    "Solo con fines educativos, no constituye asesoramiento financiero"
    "</div>",
    unsafe_allow_html=True,
)
