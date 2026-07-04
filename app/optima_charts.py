"""
charts.py
==========
Funciones de visualización con Plotly para la plataforma de optimización
de portafolios. Tema oscuro inspirado en terminales profesionales
(Bloomberg/FactSet).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TEMPLATE = "plotly_dark"

COLOR_SEQUENCE = [
    "#00D4FF",
    "#FF6B35",
    "#7CFC00",
    "#FFD700",
    "#FF4B91",
    "#8A2BE2",
    "#00FA9A",
    "#FFA500",
    "#1E90FF",
    "#FF1493",
]


def plot_price_history(prices: pd.DataFrame) -> go.Figure:
    """Línea de precios normalizados a base 100."""
    normalized = prices / prices.iloc[0] * 100
    fig = go.Figure()
    for i, col in enumerate(normalized.columns):
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[col],
                mode="lines",
                name=col,
                line=dict(color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)], width=1.8),
            )
        )
    fig.update_layout(
        template=TEMPLATE,
        title="Evolución de precios (Base 100)",
        xaxis_title="Fecha",
        yaxis_title="Precio normalizado",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
        height=420,
    )
    return fig


def plot_weights_pie(weights: pd.Series, title: str = "Asignación del Portafolio") -> go.Figure:
    w = weights[weights.abs() > 1e-4]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=w.index,
                values=w.values,
                hole=0.45,
                marker=dict(colors=COLOR_SEQUENCE),
                textinfo="label+percent",
                sort=False,
            )
        ]
    )
    fig.update_layout(
        template=TEMPLATE,
        title=title,
        margin=dict(l=10, r=10, t=60, b=10),
        height=420,
    )
    return fig


def plot_weights_bar(weights: pd.Series, title: str = "Pesos del Portafolio") -> go.Figure:
    w = weights.sort_values(ascending=True)
    colors = ["#FF4B4B" if v < 0 else "#00D4FF" for v in w.values]
    fig = go.Figure(
        data=[go.Bar(x=w.values * 100, y=w.index, orientation="h", marker_color=colors)]
    )
    fig.update_layout(
        template=TEMPLATE,
        title=title,
        xaxis_title="Peso (%)",
        margin=dict(l=10, r=10, t=60, b=10),
        height=max(300, 35 * len(w)),
    )
    return fig


def plot_efficient_frontier(
    mc_df: pd.DataFrame,
    frontier_df: pd.DataFrame,
    special_points: dict[str, tuple[float, float]] | None = None,
) -> go.Figure:
    """Nube Monte Carlo + frontera eficiente analítica + portafolios destacados."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=mc_df["Volatilidad"] * 100,
            y=mc_df["Retorno"] * 100,
            mode="markers",
            marker=dict(
                size=4,
                color=mc_df["Sharpe"],
                colorscale="Viridis",
                colorbar=dict(title="Sharpe"),
                opacity=0.5,
            ),
            name=f"Monte Carlo ({len(mc_df):,} portafolios)",
            hovertemplate="Vol: %{x:.2f}%<br>Retorno: %{y:.2f}%<extra></extra>",
        )
    )

    if not frontier_df.empty:
        fig.add_trace(
            go.Scatter(
                x=frontier_df["Volatilidad"] * 100,
                y=frontier_df["Retorno"] * 100,
                mode="lines",
                line=dict(color="#FFD700", width=3),
                name="Frontera Eficiente (Markowitz)",
            )
        )

    if special_points:
        marker_symbols = ["star", "diamond", "circle", "x", "triangle-up"]
        for i, (label, (vol, ret)) in enumerate(special_points.items()):
            fig.add_trace(
                go.Scatter(
                    x=[vol * 100],
                    y=[ret * 100],
                    mode="markers+text",
                    marker=dict(
                        size=16,
                        symbol=marker_symbols[i % len(marker_symbols)],
                        color="#FF4B4B",
                        line=dict(width=2, color="white"),
                    ),
                    text=[label],
                    textposition="top center",
                    name=label,
                )
            )

    fig.update_layout(
        template=TEMPLATE,
        title="Frontera Eficiente (Markowitz) + Nube Monte Carlo",
        xaxis_title="Volatilidad Anualizada (%)",
        yaxis_title="Retorno Anualizado (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=70, b=10),
        height=550,
    )
    return fig


def plot_equity_curve(equity: pd.Series, benchmark: pd.Series | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            name=equity.name or "Portafolio",
            line=dict(color="#00D4FF", width=2.2),
        )
    )
    if benchmark is not None:
        fig.add_trace(
            go.Scatter(
                x=benchmark.index,
                y=benchmark.values,
                mode="lines",
                name=benchmark.name or "Benchmark",
                line=dict(color="#FFA500", width=1.6, dash="dash"),
            )
        )
    fig.update_layout(
        template=TEMPLATE,
        title="Curva de Capital (Backtesting)",
        xaxis_title="Fecha",
        yaxis_title="Valor del Portafolio ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
        height=450,
    )
    return fig


def plot_drawdown(returns: pd.Series, benchmark_returns: pd.Series | None = None) -> go.Figure:
    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax() * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.values,
            mode="lines",
            fill="tozeroy",
            name=returns.name or "Portafolio",
            line=dict(color="#FF4B4B", width=1.5),
        )
    )
    if benchmark_returns is not None:
        cum_b = (1 + benchmark_returns).cumprod()
        dd_b = (cum_b - cum_b.cummax()) / cum_b.cummax() * 100
        fig.add_trace(
            go.Scatter(
                x=dd_b.index,
                y=dd_b.values,
                mode="lines",
                name=benchmark_returns.name or "Benchmark",
                line=dict(color="#FFA500", width=1.2, dash="dash"),
            )
        )
    fig.update_layout(
        template=TEMPLATE,
        title="Drawdown (%)",
        xaxis_title="Fecha",
        yaxis_title="Drawdown (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
        height=300,
    )
    return fig


def plot_weights_over_time(weights_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(weights_df.columns):
        fig.add_trace(
            go.Scatter(
                x=weights_df.index,
                y=weights_df[col] * 100,
                mode="lines",
                stackgroup="one",
                name=col,
                line=dict(width=0.5, color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)]),
            )
        )
    fig.update_layout(
        template=TEMPLATE,
        title="Evolución de los Pesos del Portafolio (Rebalanceo)",
        xaxis_title="Fecha",
        yaxis_title="Peso (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
        height=400,
    )
    return fig


def plot_stress_scenario(result: dict) -> go.Figure:
    cum_port = (result["cumulative_returns"] - 1) * 100
    cum_bench = (result["benchmark_cumulative"] - 1) * 100
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cum_port.index,
            y=cum_port.values,
            mode="lines",
            name="Portafolio Optimizado",
            line=dict(color="#00D4FF", width=2.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cum_bench.index,
            y=cum_bench.values,
            mode="lines",
            name="Equal Weight (Referencia)",
            line=dict(color="#FFA500", width=1.6, dash="dash"),
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        template=TEMPLATE,
        title=f"{result['scenario']}: {result['period_start'].date()} → {result['period_end'].date()}",
        xaxis_title="Fecha",
        yaxis_title="Retorno Acumulado (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
        height=380,
    )
    return fig


def plot_stress_summary(results: dict) -> go.Figure:
    names, port_vals, bench_vals = [], [], []
    for name, r in results.items():
        if r is None:
            continue
        names.append(name)
        port_vals.append(r["total_return"] * 100)
        bench_vals.append(r["benchmark_total_return"] * 100)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=names, y=port_vals, name="Portafolio Optimizado", marker_color="#00D4FF")
    )
    fig.add_trace(go.Bar(x=names, y=bench_vals, name="Equal Weight", marker_color="#FFA500"))
    fig.add_hline(y=0, line_color="gray")
    fig.update_layout(
        template=TEMPLATE,
        title="Retorno Total durante cada Escenario de Crisis",
        yaxis_title="Retorno Total (%)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
        height=420,
    )
    return fig


def plot_strategy_comparison(df: pd.DataFrame, metric: str = "Sharpe Ratio") -> go.Figure:
    df_sorted = df.sort_values(metric, ascending=True)
    colors = ["#00FA9A" if v >= 0 else "#FF4B4B" for v in df_sorted[metric]]
    fig = go.Figure(
        data=[go.Bar(x=df_sorted[metric], y=df_sorted.index, orientation="h", marker_color=colors)]
    )
    fig.update_layout(
        template=TEMPLATE,
        title=f"Comparación de Estrategias: {metric}",
        xaxis_title=metric,
        margin=dict(l=10, r=10, t=60, b=10),
        height=max(400, 32 * len(df_sorted)),
    )
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            colorbar=dict(title="Correlación"),
        )
    )
    fig.update_layout(
        template=TEMPLATE,
        title="Matriz de Correlación",
        margin=dict(l=10, r=10, t=60, b=10),
        height=450,
    )
    return fig
