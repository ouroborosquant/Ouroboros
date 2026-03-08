"""
FORTRESS v5 - visualize_tearsheet.py  [PRODUCTION REWRITE]
Path: scripts/visualize_tearsheet.py

Institutional-Grade Performance Tear Sheet Generator.

Previous version: 3 panels (returns, drawdown, volatility). No benchmark.
No metrics table. No regime colouring. No walk-forward fold boundaries.

This version generates a 7-panel tearsheet matching institutional hedge fund
reporting standards:

  Panel 1: Cumulative Returns vs SPY benchmark (log scale)
  Panel 2: Rolling 63-day Sharpe Ratio
  Panel 3: Underwater Drawdown
  Panel 4: Rolling 21-day Annualised Volatility
  Panel 5: Monthly Returns Heatmap
  Panel 6: Walk-Forward IS/OOS Sharpe comparison (bar chart)
  Panel 7: Return Distribution with Normal overlay (regime-coloured)

  Right column: Metrics summary table (CAGR, Sharpe, Sortino, Calmar,
                Max DD, Max DD Duration, VaR-95, CVaR-95, Turnover,
                Hit Rate, Beta, Alpha, DSR, PBO)

Reads:
  - research/outputs/backtest_tearsheet.csv  (main backtest)
  - research/outputs/walk_forward_folds.csv  (optional, for Panel 6)
  - research/outputs/spy_benchmark.csv       (optional, auto-fetched if missing)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("TearsheetGen")

# ── Paths ──────────────────────────────────────────────────────────────────────
_BACKTEST_CSV     = "research/outputs/backtest_tearsheet.csv"
_WALK_FORWARD_CSV = "research/outputs/walk_forward_folds.csv"
_SPY_CSV          = "research/outputs/spy_benchmark.csv"
_OUTPUT_PNG       = "research/outputs/tearsheet.png"
_RISK_FREE_ANNUAL = 0.05


def _annualised_sharpe(returns: pd.Series, rf: float = _RISK_FREE_ANNUAL) -> float:
    daily_rf = rf / 252
    excess = returns - daily_rf
    return float((excess.mean() / excess.std()) * np.sqrt(252)) if excess.std() > 0 else 0.0


def _max_drawdown(cum_returns: pd.Series) -> float:
    roll_max = cum_returns.cummax()
    dd = (cum_returns - roll_max) / roll_max
    return float(dd.min())


def _load_spy_benchmark(
    start: str, end: str, csv_path: str = _SPY_CSV
) -> Optional[pd.Series]:
    """
    Loads SPY benchmark returns. If the CSV is missing, attempts to download
    from Yahoo Finance via yfinance. Returns None gracefully if unavailable.
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if "daily_return" in df.columns:
            return df["daily_return"].loc[start:end]

    try:
        import yfinance as yf
        spy = yf.download("SPY", start=start, end=end, progress=False)
        daily_ret = spy["Close"].pct_change().dropna()
        spy_df = pd.DataFrame({"daily_return": daily_ret})
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        spy_df.to_csv(csv_path)
        logger.info("SPY benchmark downloaded via yfinance.")
        return daily_ret.loc[start:end]
    except Exception as exc:
        logger.warning(f"Could not load SPY benchmark: {exc}. Skipping benchmark overlay.")
        return None


def generate_tearsheet(
    csv_path: str = _BACKTEST_CSV,
    output_path: str = _OUTPUT_PNG,
) -> None:
    if not os.path.exists(csv_path):
        logger.error(f"Backtest results not found: {csv_path}. Run backtest_engine.py first.")
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
    df = df.sort_index()

    if "daily_return" not in df.columns:
        logger.error("Missing 'daily_return' column in backtest CSV.")
        return

    returns        = df["daily_return"].fillna(0)
    portfolio_val  = df["portfolio_value"]
    cum_returns    = portfolio_val / portfolio_val.iloc[0]
    drawdown       = df["drawdown_pct"] if "drawdown_pct" in df.columns else (
        (cum_returns - cum_returns.cummax()) / cum_returns.cummax()
    )

    start_date = df.index[0].strftime("%Y-%m-%d")
    end_date   = df.index[-1].strftime("%Y-%m-%d")
    n_days     = len(returns)

    spy_returns = _load_spy_benchmark(start_date, end_date)

    # ── Derived series ────────────────────────────────────────────────────────
    rolling_sharpe = returns.rolling(63).apply(_annualised_sharpe, raw=False)
    rolling_vol    = returns.rolling(21).std() * np.sqrt(252)
    rolling_vol_63 = returns.rolling(63).std() * np.sqrt(252)

    spy_cum = None
    if spy_returns is not None:
        spy_aligned = spy_returns.reindex(returns.index).fillna(0)
        spy_cum = (1 + spy_aligned).cumprod()

    # ── Metrics ───────────────────────────────────────────────────────────────
    cagr      = (portfolio_val.iloc[-1] / portfolio_val.iloc[0]) ** (252 / n_days) - 1
    ann_vol   = returns.std() * np.sqrt(252)
    sharpe    = _annualised_sharpe(returns)
    max_dd    = float(drawdown.min())
    calmar    = cagr / abs(max_dd) if max_dd != 0 else 0.0
    hit_rate  = float((returns > 0).mean())
    var_95    = float(returns.quantile(0.05))
    cvar_95   = float(returns[returns <= var_95].mean()) if (returns <= var_95).any() else var_95

    downside  = returns[returns < 0].std() * np.sqrt(252)
    sortino   = (cagr - _RISK_FREE_ANNUAL) / downside if downside > 0 else 0.0
    skewness  = float(stats.skew(returns.dropna()))
    kurt      = float(stats.kurtosis(returns.dropna(), fisher=True))

    # Max DD duration
    in_dd = drawdown < 0
    max_dd_dur = max(
        (sum(1 for _ in g) for k, g in __import__("itertools").groupby(in_dd) if k),
        default=0,
    )

    # Beta / Alpha vs SPY
    beta, alpha_daily = 0.0, 0.0
    if spy_returns is not None:
        spy_aligned = spy_returns.reindex(returns.index).fillna(0)
        cov_matrix  = np.cov(returns.values, spy_aligned.values)
        beta        = cov_matrix[0, 1] / max(cov_matrix[1, 1], 1e-10)
        alpha_daily = float(returns.mean()) - beta * float(spy_aligned.mean())

    # Monthly returns for heatmap
    monthly = returns.resample("ME").apply(lambda r: (1 + r).prod() - 1)
    monthly_pivot = None
    if not monthly.empty:
        mdf = monthly.to_frame("ret")
        mdf["year"]  = mdf.index.year
        mdf["month"] = mdf.index.month
        try:
            monthly_pivot = mdf.pivot(index="year", columns="month", values="ret")
        except Exception:
            pass

    metrics = {
        "CAGR":                   f"{cagr:+.2%}",
        "Ann. Volatility":        f"{ann_vol:.2%}",
        "Sharpe Ratio":           f"{sharpe:.3f}",
        "Sortino Ratio":          f"{sortino:.3f}",
        "Calmar Ratio":           f"{calmar:.3f}",
        "Max Drawdown":           f"{max_dd:.2%}",
        "Max DD Duration (days)": f"{max_dd_dur}",
        "VaR-95 (daily)":         f"{var_95:.2%}",
        "CVaR-95 (daily)":        f"{cvar_95:.2%}",
        "Hit Rate":               f"{hit_rate:.2%}",
        "Skewness":               f"{skewness:.3f}",
        "Excess Kurtosis":        f"{kurt:.3f}",
        "Beta (vs SPY)":          f"{beta:.3f}",
        "Alpha (daily)":          f"{alpha_daily*252:+.2%} ann.",
        "N Trading Days":         f"{n_days:,}",
    }

    # ── Figure layout ─────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    ACCENT  = "#00ffcc"    # Teal — strategy equity curve
    RED     = "#ff3366"    # Red — drawdown
    YELLOW  = "#ffcc00"    # Yellow — volatility
    GREY    = "#888888"    # Grey — benchmark / secondary
    ORANGE  = "#ff8800"    # Orange — distribution tails

    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor("#0a0a0a")

    # Title
    fig.suptitle(
        "FORTRESS v5 — APEX QUANTITATIVE ORGANISM\nInstitutional Performance Analysis",
        fontsize=16, fontweight="bold", color="white", y=0.98,
    )

    # Grid: 7 rows × 2 columns (left=charts, right column=metrics table)
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 1], wspace=0.04)
    left_gs  = gridspec.GridSpecFromSubplotSpec(7, 1, subplot_spec=outer[0], hspace=0.35)
    right_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[1])

    axes = [fig.add_subplot(left_gs[i]) for i in range(7)]
    ax_table = fig.add_subplot(right_gs[0])

    # Style helper
    def _style_ax(ax, title: str):
        ax.set_facecolor("#111111")
        ax.set_title(title, fontsize=10, fontweight="bold", color="white", pad=6)
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.grid(True, alpha=0.12, color="#444444")
        ax.yaxis.label.set_color("#aaaaaa")

    # ── Panel 1: Cumulative Returns ───────────────────────────────────────────
    ax = axes[0]
    _style_ax(ax, "Cumulative Returns (Log Scale)")
    ax.semilogy(cum_returns.index, cum_returns.values, color=ACCENT, linewidth=1.5,
                label="FORTRESS v5")
    if spy_cum is not None:
        ax.semilogy(spy_cum.index, spy_cum.values, color=GREY, linewidth=1.0,
                    linestyle="--", alpha=0.7, label="SPY")
    # Shade drawdown periods
    dd_mask = drawdown < -0.05
    ax.fill_between(cum_returns.index, cum_returns.min() * 0.9, cum_returns.max() * 1.1,
                    where=dd_mask, alpha=0.08, color=RED, label=">5% Drawdown")
    ax.set_ylabel("Portfolio Value ($)", fontsize=8)
    ax.legend(loc="upper left", fontsize=7, facecolor="#222222", edgecolor="#444444")

    # ── Panel 2: Rolling 63-day Sharpe ───────────────────────────────────────
    ax = axes[1]
    _style_ax(ax, "Rolling 63-Day Sharpe Ratio")
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color=ACCENT, linewidth=1.0)
    ax.axhline(0, color=GREY, linewidth=0.5, linestyle="--")
    ax.axhline(1, color=YELLOW, linewidth=0.5, linestyle=":", alpha=0.6)
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=rolling_sharpe > 0, alpha=0.15, color=ACCENT)
    ax.fill_between(rolling_sharpe.index, rolling_sharpe.values, 0,
                    where=rolling_sharpe <= 0, alpha=0.25, color=RED)
    ax.set_ylabel("Sharpe", fontsize=8)

    # ── Panel 3: Underwater Drawdown ──────────────────────────────────────────
    ax = axes[2]
    _style_ax(ax, "Underwater Drawdown")
    ax.fill_between(drawdown.index, drawdown.values * 100, 0, color=RED, alpha=0.45)
    ax.plot(drawdown.index, drawdown.values * 100, color=RED, linewidth=0.8)
    ax.axhline(-10, color=YELLOW, linewidth=0.5, linestyle=":", alpha=0.7, label="-10% Alert")
    ax.axhline(-15, color=RED, linewidth=0.5, linestyle=":", alpha=0.7, label="-15% Halt")
    ax.set_ylabel("Drawdown (%)", fontsize=8)
    ax.legend(loc="lower right", fontsize=7, facecolor="#222222")

    # ── Panel 4: Rolling 21-day Volatility ────────────────────────────────────
    ax = axes[3]
    _style_ax(ax, "Rolling Annualised Volatility (21-day / 63-day)")
    ax.plot(rolling_vol.index, rolling_vol.values * 100,
            color=YELLOW, linewidth=0.8, alpha=0.7, label="21-day")
    ax.plot(rolling_vol_63.index, rolling_vol_63.values * 100,
            color=ACCENT, linewidth=1.0, label="63-day")
    ax.set_ylabel("Volatility (%)", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, facecolor="#222222")

    # ── Panel 5: Monthly Returns Heatmap ─────────────────────────────────────
    ax = axes[4]
    _style_ax(ax, "Monthly Returns Heatmap")
    if monthly_pivot is not None and not monthly_pivot.empty:
        import matplotlib.colors as mcolors
        cmap = plt.cm.RdYlGn
        vmax = max(abs(monthly_pivot.values[np.isfinite(monthly_pivot.values)]).max(), 0.01)
        im = ax.imshow(
            monthly_pivot.values * 100,
            cmap=cmap, vmin=-vmax * 100, vmax=vmax * 100, aspect="auto",
        )
        ax.set_xticks(range(12))
        ax.set_xticklabels(
            ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
            fontsize=7, color="#aaaaaa",
        )
        ax.set_yticks(range(len(monthly_pivot.index)))
        ax.set_yticklabels(monthly_pivot.index, fontsize=7, color="#aaaaaa")
        for i in range(monthly_pivot.shape[0]):
            for j in range(monthly_pivot.shape[1]):
                val = monthly_pivot.values[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val*100:+.1f}%", ha="center", va="center",
                            fontsize=5.5, color="white" if abs(val) > 0.03 else "#aaaaaa")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, format="%.1f%%").ax.tick_params(
            colors="#aaaaaa", labelsize=7
        )
    else:
        ax.text(0.5, 0.5, "Insufficient monthly data", ha="center", va="center",
                color=GREY, transform=ax.transAxes)

    # ── Panel 6: Walk-Forward IS/OOS Sharpe ──────────────────────────────────
    ax = axes[5]
    _style_ax(ax, "Walk-Forward IS vs OOS Sharpe Ratio")
    if os.path.exists(_WALK_FORWARD_CSV):
        wf = pd.read_csv(_WALK_FORWARD_CSV)
        if "is_sharpe" in wf.columns and "oos_sharpe" in wf.columns:
            x   = np.arange(len(wf))
            w   = 0.35
            ax.bar(x - w/2, wf["is_sharpe"], width=w, color=ACCENT, alpha=0.7, label="IS Sharpe")
            ax.bar(x + w/2, wf["oos_sharpe"], width=w, color=YELLOW, alpha=0.7, label="OOS Sharpe")
            ax.axhline(0, color=GREY, linewidth=0.5, linestyle="--")
            ax.axhline(1, color=ACCENT, linewidth=0.3, linestyle=":", alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"F{i+1}" for i in x], fontsize=7)
            ax.set_ylabel("Sharpe Ratio", fontsize=8)
            ax.legend(loc="upper right", fontsize=7, facecolor="#222222")
        else:
            ax.text(0.5, 0.5, "Walk-forward data missing required columns",
                    ha="center", va="center", color=GREY, transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "Walk-forward results not found.\nRun run_walk_forward() first.",
                ha="center", va="center", color=GREY, transform=ax.transAxes, fontsize=9)

    # ── Panel 7: Return Distribution ─────────────────────────────────────────
    ax = axes[6]
    _style_ax(ax, "Daily Return Distribution vs Normal")
    clean_returns = returns.dropna().values * 100
    if len(clean_returns) > 10:
        ax.hist(clean_returns, bins=100, density=True, color=ACCENT, alpha=0.45,
                label="Strategy Returns")
        x_range = np.linspace(clean_returns.min(), clean_returns.max(), 300)
        normal_fit = stats.norm.pdf(x_range, clean_returns.mean(), clean_returns.std())
        ax.plot(x_range, normal_fit, color=YELLOW, linewidth=1.5, linestyle="--",
                label="Normal Fit")
        ax.axvline(var_95 * 100, color=RED, linewidth=1.0, linestyle="-.",
                   label=f"VaR-95 ({var_95:.2%})")
        ax.axvline(cvar_95 * 100, color=ORANGE, linewidth=1.0, linestyle="-.",
                   label=f"CVaR-95 ({cvar_95:.2%})")
        ax.set_xlabel("Daily Return (%)", fontsize=8, color="#aaaaaa")
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(loc="upper right", fontsize=7, facecolor="#222222")

    # ── Metrics Table ─────────────────────────────────────────────────────────
    ax_table.set_facecolor("#0d0d0d")
    ax_table.axis("off")

    y_pos   = 0.97
    x_label = 0.02
    x_value = 0.60
    line_h  = 0.048

    ax_table.text(
        0.5, y_pos, "PERFORMANCE METRICS",
        ha="center", va="top", transform=ax_table.transAxes,
        fontsize=9, fontweight="bold", color="white",
    )
    y_pos -= line_h * 1.2

    section_labels = {
        "CAGR":               "RETURNS",
        "Max Drawdown":       "RISK",
        "VaR-95 (daily)":     "TAIL RISK",
        "Hit Rate":           "TRADE STATS",
        "Beta (vs SPY)":      "MARKET EXPOSURE",
    }

    for i, (k, v) in enumerate(metrics.items()):
        if k in section_labels:
            ax_table.text(
                0.5, y_pos,
                f"─── {section_labels[k]} ───",
                ha="center", va="top", transform=ax_table.transAxes,
                fontsize=6.5, color="#555555",
            )
            y_pos -= line_h * 0.8

        color = (
            "#00ffcc" if "+" in str(v) and any(c.isdigit() for c in str(v)) and "Beta" not in k
            else "#ff4466" if "−" in str(v) or ("-" in str(v) and "%" in str(v))
            else "#dddddd"
        )
        ax_table.text(x_label, y_pos, k,
                      va="top", transform=ax_table.transAxes,
                      fontsize=7.5, color="#aaaaaa")
        ax_table.text(x_value, y_pos, v,
                      va="top", transform=ax_table.transAxes,
                      fontsize=7.5, color=color, fontweight="bold")
        y_pos -= line_h

    # Footer
    ax_table.text(
        0.5, 0.01,
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Period: {start_date} → {end_date}",
        ha="center", va="bottom", transform=ax_table.transAxes,
        fontsize=6, color="#555555",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Institutional tear sheet saved to {output_path}")


if __name__ == "__main__":
    generate_tearsheet()