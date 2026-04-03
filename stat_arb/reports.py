from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots

from .backtests import BacktestResult, MeanReversionConfig


def _format_stat(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(numeric):
        return "-"
    if abs(numeric) >= 1000:
        return f"{numeric:,.2f}"
    if abs(numeric) >= 1:
        return f"{numeric:.3f}"
    return f"{numeric:.4f}"


def _render_weights_table(weights: dict[str, float]) -> str:
    rows = []
    for symbol, weight in weights.items():
        side = "Long" if weight > 0 else "Short"
        rows.append(
            f"<tr><td>{escape(symbol)}</td><td>{side}</td><td>{abs(weight):.3f}</td></tr>"
        )
    return "\n".join(rows)


def _render_stats_grid(stats: pd.Series, columns: Iterable[str]) -> str:
    cards = []
    for column in columns:
        if column not in stats.index:
            continue
        cards.append(
            "<div class='stat-card'>"
            f"<div class='stat-label'>{escape(column.replace('_', ' ').title())}</div>"
            f"<div class='stat-value'>{escape(_format_stat(stats[column]))}</div>"
            "</div>"
        )
    return "\n".join(cards)


def _normalized_prices(prices: pd.DataFrame) -> pd.DataFrame:
    base = prices.iloc[0].replace(0, pd.NA)
    return prices.divide(base).multiply(100.0)


def _build_backtest_figure(
    result: BacktestResult, config: MeanReversionConfig | None
) -> go.Figure:
    has_signal = {"spread_z", "position"}.issubset(result.timeseries.columns)
    rows = 4 if has_signal else 3
    titles = [
        "Normalized Component Prices",
        "Basket Notional",
        "Spread Z-Score" if has_signal else "Portfolio Returns",
        "Position and Equity Curve" if has_signal else "Equity Curve",
    ]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=titles[:rows],
        specs=[[{"secondary_y": False}] for _ in range(rows)],
    )

    normalized = _normalized_prices(result.prices)
    for symbol in normalized.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[symbol],
                mode="lines",
                name=f"{symbol} price",
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra>" + symbol + "</extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=result.timeseries.index,
            y=result.timeseries["basket_notional"],
            mode="lines",
            name="Basket notional",
            line={"width": 2},
        ),
        row=2,
        col=1,
    )

    if has_signal:
        fig.add_trace(
            go.Scatter(
                x=result.timeseries.index,
                y=result.timeseries["spread_z"],
                mode="lines",
                name="Spread z-score",
            ),
            row=3,
            col=1,
        )
        if config is not None:
            for label, level, dash in [
                ("Entry", config.entry_z, "dash"),
                ("-Entry", -config.entry_z, "dash"),
                ("Exit", config.exit_z, "dot"),
            ]:
                fig.add_hline(
                    y=level,
                    row=3,
                    col=1,
                    line_dash=dash,
                    line_color="#8f8f8f",
                    annotation_text=label,
                    annotation_position="top left",
                )
        fig.add_trace(
            go.Scatter(
                x=result.timeseries.index,
                y=result.timeseries["position"],
                mode="lines",
                name="Position",
                line={"shape": "hv"},
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=result.timeseries.index,
                y=result.timeseries["equity_curve"],
                mode="lines",
                name="Equity curve",
                line={"width": 2},
            ),
            row=4,
            col=1,
        )
    else:
        fig.add_trace(
            go.Bar(
                x=result.timeseries.index,
                y=result.timeseries["portfolio_returns"],
                name="Daily returns",
                marker={"color": "#C96F4A"},
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=result.timeseries.index,
                y=result.timeseries["equity_curve"],
                mode="lines",
                name="Equity curve",
                line={"width": 2},
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        height=1100 if has_signal else 900,
        template="plotly_white",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        margin={"l": 50, "r": 30, "t": 70, "b": 40},
    )
    fig.update_yaxes(title_text="Indexed", row=1, col=1)
    fig.update_yaxes(title_text="Notional", row=2, col=1)
    if has_signal:
        fig.update_yaxes(title_text="Z", row=3, col=1)
        fig.update_yaxes(title_text="Position / Equity", row=4, col=1)
    else:
        fig.update_yaxes(title_text="Return / Equity", row=3, col=1)
    return fig


def _build_batch_dashboard(summary: pd.DataFrame) -> go.Figure:
    summary = summary.copy()
    if "active_fraction" not in summary.columns:
        summary["active_fraction"] = 0.0
    figure = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Risk / Return Map", "Sharpe Ratio by Trade"),
        vertical_spacing=0.12,
    )
    hover = (
        summary["trade"] if "trade" in summary.columns else summary.index.astype(str)
    )
    figure.add_trace(
        go.Scatter(
            x=summary["annual_vol"],
            y=summary["annual_return"],
            mode="markers",
            text=hover,
            marker={
                "size": summary["active_fraction"].fillna(0).mul(40).add(12),
                "color": summary["sharpe"],
                "colorscale": "RdYlGn",
                "showscale": True,
                "colorbar": {"title": "Sharpe"},
            },
            hovertemplate="%{text}<br>Vol=%{x:.3f}<br>Return=%{y:.3f}<br>Sharpe=%{marker.color:.3f}<extra></extra>",
            name="Trades",
        ),
        row=1,
        col=1,
    )

    ordered = summary.sort_values("sharpe", ascending=False).head(15)
    figure.add_trace(
        go.Bar(
            x=ordered["trade"],
            y=ordered["sharpe"],
            marker={"color": "#2D6A4F"},
            name="Sharpe",
            hovertemplate="%{x}<br>Sharpe=%{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    figure.update_layout(
        height=900,
        template="plotly_white",
        showlegend=False,
        margin={"l": 50, "r": 30, "t": 70, "b": 120},
    )
    figure.update_xaxes(title_text="Annualized Volatility", row=1, col=1)
    figure.update_yaxes(title_text="Annualized Return", row=1, col=1)
    figure.update_xaxes(tickangle=-35, row=2, col=1)
    figure.update_yaxes(title_text="Sharpe Ratio", row=2, col=1)
    return figure


def write_backtest_report(
    result: BacktestResult,
    output_path: Path,
    title: str,
    config: MeanReversionConfig | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_html = to_html(
        _build_backtest_figure(result, config),
        include_plotlyjs=True,
        full_html=False,
        config={"displaylogo": False, "responsive": True},
    )
    stats_html = _render_stats_grid(
        result.stats,
        [
            "total_return",
            "annual_return",
            "annual_vol",
            "sharpe",
            "max_drawdown",
            "win_rate",
            "entries",
            "active_fraction",
            "spread_z_last",
            "basket_last_zscore",
        ],
    )
    basket_title = escape(result.basket_label or "Custom basket")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --panel: #fffdf8;
      --ink: #1f2933;
      --muted: #6b7280;
      --accent: #2d6a4f;
      --border: #d9d2c3;
    }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(45,106,79,0.16), transparent 28%),
        linear-gradient(180deg, #f7f4ec 0%, #efe8da 100%);
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero {{
      display: grid;
      gap: 12px;
      margin-bottom: 28px;
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 0.9rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 1.05;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      max-width: 70ch;
    }}
    .panel {{
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 20px;
      box-shadow: 0 18px 60px rgba(31, 41, 51, 0.08);
      margin-bottom: 20px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
    }}
    .stat-card {{
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 14px;
      background: rgba(255,255,255,0.7);
    }}
    .stat-label {{
      color: var(--muted);
      font-size: 0.86rem;
      margin-bottom: 6px;
    }}
    .stat-value {{
      font-size: 1.35rem;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--border);
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-size: 0.88rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="eyebrow">{escape(result.strategy.replace("_", " "))}</div>
      <h1>{escape(title)}</h1>
      <p class="subtitle">{basket_title}</p>
    </section>
    <section class="panel">
      <div class="stats">{stats_html}</div>
    </section>
    <section class="panel">
      <h2>Weights</h2>
      <table>
        <thead><tr><th>Symbol</th><th>Side</th><th>Units</th></tr></thead>
        <tbody>{_render_weights_table(result.weights)}</tbody>
      </table>
    </section>
    <section class="panel">
      {figure_html}
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def write_batch_dashboard(
    summary: pd.DataFrame, output_path: Path, title: str = "Batch Backtest Dashboard"
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_html = to_html(
        _build_batch_dashboard(summary),
        include_plotlyjs=True,
        full_html=False,
        config={"displaylogo": False, "responsive": True},
    )
    table_rows = []
    preferred_columns = [
        "trade",
        "annual_return",
        "annual_vol",
        "sharpe",
        "max_drawdown",
        "entries",
        "active_fraction",
        "report_path",
    ]
    columns = [column for column in preferred_columns if column in summary.columns]
    for _, row in summary.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if column == "report_path" and isinstance(value, str):
                link = escape(value)
                cells.append(f"<td><a href='{link}'>{link}</a></td>")
            else:
                cells.append(f"<td>{escape(_format_stat(value))}</td>")
        table_rows.append("<tr>" + "".join(cells) + "</tr>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: "Palatino Linotype", "Book Antiqua", serif;
      background: linear-gradient(180deg, #f6f0e6 0%, #ebe1cf 100%);
      color: #1f2933;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    .panel {{
      background: rgba(255, 252, 245, 0.94);
      border: 1px solid #d6cab8;
      border-radius: 18px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 16px 48px rgba(31, 41, 51, 0.08);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid #e2d8c8;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: #5f6b76;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-size: 0.82rem;
    }}
    a {{
      color: #2d6a4f;
      text-decoration: none;
    }}
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>{escape(title)}</h1>
      <p>Interactive overview of batch mean-reversion runs. Bubble size reflects time in market; color reflects Sharpe ratio.</p>
      {figure_html}
    </section>
    <section class="panel">
      <h2>Run Summary</h2>
      <table>
        <thead><tr>{"".join(f"<th>{escape(column.replace('_', ' ').title())}</th>" for column in columns)}</tr></thead>
        <tbody>{"".join(table_rows)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def write_walk_forward_report(
    timeseries: pd.DataFrame,
    fold_summary: pd.DataFrame,
    stats: pd.Series,
    output_path: Path,
    title: str = "Walk-Forward Backtest",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Out-of-Sample Equity Curve", "Selected Strategies per Fold"),
    )
    fig.add_trace(
        go.Scatter(
            x=timeseries.index,
            y=timeseries["equity_curve"],
            mode="lines",
            name="Equity curve",
            line={"color": "#2D6A4F", "width": 2},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=timeseries.index,
            y=timeseries["selected_count"],
            name="Selected baskets",
            marker={"color": "#C96F4A"},
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=780,
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        margin={"l": 50, "r": 30, "t": 70, "b": 40},
    )
    fig.update_yaxes(title_text="Growth of $1", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    figure_html = to_html(
        fig,
        include_plotlyjs=True,
        full_html=False,
        config={"displaylogo": False, "responsive": True},
    )
    stats_html = _render_stats_grid(
        stats,
        [
            "total_return",
            "annual_return",
            "annual_vol",
            "sharpe",
            "max_drawdown",
            "win_rate",
        ],
    )

    fold_columns = [
        "fold",
        "rank",
        "trade",
        "selection_score",
        "train_sharpe",
        "validation_sharpe",
        "test_sharpe",
        "validation_entries",
        "test_entries",
        "test_start",
        "test_end",
    ]
    columns = [column for column in fold_columns if column in fold_summary.columns]
    rows = []
    for _, row in fold_summary.iterrows():
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{escape(_format_stat(row[column]))}</td>" for column in columns
            )
            + "</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: #1f2933;
      background: linear-gradient(180deg, #f7f2e8 0%, #ece2d2 100%);
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }}
    .panel {{
      background: rgba(255,253,248,0.94);
      border: 1px solid #dacfbf;
      border-radius: 18px;
      padding: 20px;
      margin-bottom: 20px;
      box-shadow: 0 18px 56px rgba(31, 41, 51, 0.08);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
    }}
    .stat-card {{
      border: 1px solid #dacfbf;
      border-radius: 14px;
      padding: 14px;
      background: rgba(255,255,255,0.72);
    }}
    .stat-label {{
      color: #66717d;
      font-size: 0.86rem;
      margin-bottom: 6px;
    }}
    .stat-value {{
      font-size: 1.35rem;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid #e3d9ca;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: #5f6b76;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-size: 0.82rem;
    }}
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>{escape(title)}</h1>
      <p>Walk-forward selection uses train and validation windows to choose baskets, then reports only the next out-of-sample test segment.</p>
    </section>
    <section class="panel">
      <div class="stats">{stats_html}</div>
    </section>
    <section class="panel">
      {figure_html}
    </section>
    <section class="panel">
      <h2>Fold Summary</h2>
      <table>
        <thead><tr>{"".join(f"<th>{escape(column.replace('_', ' ').title())}</th>" for column in columns)}</tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
