"""
scripts/visualize_v5_comparison.py
────────────────────────────────────
Renders the v5 vs baseline comparison as a 6-panel dashboard.

Reads:
  research/outputs/v5_wf_results.csv
  research/outputs/wf_folds.csv          (v8.2 baseline, optional)
  research/outputs/v5_tearsheet.csv

Writes:
  research/outputs/v5_comparison.png

Panels:
  1. OOS Sharpe ratio per fold  (v5 bars vs baseline markers)
  2. OOS Sortino ratio per fold
  3. OOS Max Drawdown per fold  (smaller = better)
  4. OOS CAGR per fold
  5. Avg daily turnover per fold (v5 improvement in friction)
  6. Cumulative OOS return (stitched, v5 vs baseline)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OUT_DIR   = Path("research/outputs")
_V5_WF     = _OUT_DIR / "v5_wf_results.csv"
_BASE_WF   = _OUT_DIR / "wf_folds.csv"
_V5_TS     = _OUT_DIR / "v5_tearsheet.csv"
_PLOT_OUT  = _OUT_DIR / "v5_comparison.png"


def _annualize_returns(daily_ret: pd.Series) -> pd.Series:
    return (1.0 + daily_ret).cumprod()


def _max_dd_series(cum_ret: pd.Series) -> pd.Series:
    peak = cum_ret.cummax()
    return (cum_ret - peak) / (peak + 1e-10)


def main() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.ticker as mtick
    except ImportError:
        logger.error("matplotlib not available — skipping visualization")
        return

    if not _V5_WF.exists():
        logger.error("v5_wf_results.csv not found. Run run_v5_backtest.py first.")
        return

    v5   = pd.read_csv(_V5_WF).set_index("fold_id")
    base = pd.read_csv(_BASE_WF).set_index("fold_id") if _BASE_WF.exists() else None

    folds = sorted(v5.index.tolist())
    x     = np.arange(len(folds))
    labels = [f"F{fid}" for fid in folds]

    # ── Figure setup ──────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    BG   = "#0d0d0d"
    CARD = "#141414"
    V5_C = "#00e5cc"       # teal — v5
    BASE_C = "#ff6b6b"     # coral — baseline
    NEU  = "#888888"
    GRID = "#222222"

    fig = plt.figure(figsize=(22, 20), facecolor=BG)
    fig.suptitle(
        "FORTRESS v5 — Architectural Upgrades vs v8.2 Baseline\n"
        "OOS Walk-Forward Performance  (9 Folds · 2019-2023)",
        fontsize=15, fontweight="bold", color="white", y=0.99
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.28)

    def _ax(pos, title: str, ylabel: str = ""):
        ax = fig.add_subplot(pos)
        ax.set_facecolor(CARD)
        ax.set_title(title, fontsize=11, fontweight="bold", color="white", pad=8)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9, color=NEU)
        ax.tick_params(colors=NEU, labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a2a")
        ax.grid(True, alpha=0.15, color=GRID, linewidth=0.6)
        return ax

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _bar_compare(ax, metric_v5: str, metric_b: str, title: str,
                     ylabel: str, pct: bool = False, invert: bool = False):
        """Bar chart: v5 side-by-side with baseline."""
        v5_vals   = v5[metric_v5].reindex(folds).values
        base_vals = base[metric_b].reindex(folds).values if base is not None and metric_b in base else None

        width = 0.35
        bars_v5 = ax.bar(x - width/2, v5_vals, width, color=V5_C, alpha=0.85, label="v5", zorder=3)

        if base_vals is not None:
            bars_b = ax.bar(x + width/2, base_vals, width, color=BASE_C, alpha=0.70, label="v8.2", zorder=3)

        ax.axhline(0, color=NEU, linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, color="white")
        ax.set_title(title, fontsize=11, fontweight="bold", color="white", pad=8)
        ax.set_ylabel(ylabel, fontsize=9, color=NEU)
        ax.set_facecolor(CARD)
        for sp in ax.spines.values(): sp.set_edgecolor("#2a2a2a")
        ax.grid(True, alpha=0.15, color=GRID, linewidth=0.6)
        ax.legend(fontsize=8, loc="upper left", facecolor="#1a1a1a", edgecolor="#333")

        # Value annotations
        for i, (bar, val) in enumerate(zip(bars_v5, v5_vals)):
            fmt = f"{val:.2%}" if pct else f"{val:+.2f}"
            color = V5_C if (not invert and val > 0) or (invert and val < 0) else BASE_C
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    fmt, ha="center", va="bottom", fontsize=7, color=color, fontweight="bold")

    # ── Panel 1: OOS Sharpe ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _bar_compare(ax1, "oos_sharpe", "oos_sharpe", "OOS Sharpe Ratio", "Sharpe")

    # Target Sharpe line
    ax1.axhline(0.94, color="#ffcc00", linewidth=1.2, linestyle=":", alpha=0.7)
    ax1.text(len(folds) - 0.5, 0.96, "v8.2 avg (0.94)", fontsize=7, color="#ffcc00")

    # ── Panel 2: OOS Sortino ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _bar_compare(ax2, "oos_sortino", "oos_sortino", "OOS Sortino Ratio", "Sortino")

    # ── Panel 3: OOS Max Drawdown ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _bar_compare(ax3, "oos_max_dd", "oos_max_dd",
                 "OOS Max Drawdown  (closer to 0 = better)", "Max DD", pct=True, invert=True)

    # Prop firm kill line
    ax3.axhline(-0.08, color="#ff3333", linewidth=1.5, linestyle="--", alpha=0.9)
    ax3.text(0.01, -0.075, "8% kill limit", transform=ax3.get_yaxis_transform(),
             fontsize=8, color="#ff3333", alpha=0.9)

    # ── Panel 4: OOS CAGR ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _bar_compare(ax4, "oos_cagr", "oos_cagr", "OOS CAGR", "CAGR", pct=True)

    # ── Panel 5: Avg Turnover ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])

    v5_turn   = v5["avg_turnover"].reindex(folds).values * 100
    base_turn = base["avg_turnover"].reindex(folds).values * 100 if base is not None and "avg_turnover" in base else None

    bars = ax5.bar(x - 0.175, v5_turn,   0.35, color=V5_C,   alpha=0.85, label="v5  (CVaR+γ)", zorder=3)
    if base_turn is not None:
        ax5.bar(x + 0.175, base_turn, 0.35, color=BASE_C, alpha=0.70, label="v8.2 (MVO)",   zorder=3)

    ax5.set_xticks(x); ax5.set_xticklabels(labels, fontsize=9, color="white")
    ax5.set_title("Avg Daily Turnover  (γ-penalty reduction)", fontsize=11,
                  fontweight="bold", color="white", pad=8)
    ax5.set_ylabel("Turnover %", fontsize=9, color=NEU)
    ax5.set_facecolor(CARD)
    for sp in ax5.spines.values(): sp.set_edgecolor("#2a2a2a")
    ax5.grid(True, alpha=0.15, color=GRID)
    ax5.legend(fontsize=8, loc="upper right", facecolor="#1a1a1a", edgecolor="#333")
    ax5.yaxis.set_major_formatter(mtick.PercentFormatter())

    # ── Panel 6: Cumulative OOS equity curve ─────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])

    if _V5_TS.exists():
        ts_v5 = pd.read_csv(_V5_TS, index_col=0, parse_dates=True)["daily_return"].dropna()
        cum_v5 = _annualize_returns(ts_v5)
        ax6.plot(cum_v5.index, cum_v5.values, color=V5_C, linewidth=1.5, label="v5", zorder=4)

        # Drawdown fill
        dd_v5 = _max_dd_series(cum_v5)
        ax6_twin = ax6.twinx()
        ax6_twin.fill_between(dd_v5.index, dd_v5.values, 0,
                              color="#ff3333", alpha=0.12, label="DD")
        ax6_twin.axhline(-0.08, color="#ff3333", linewidth=1.0, linestyle="--", alpha=0.6)
        ax6_twin.set_ylim(-0.25, 0.05)
        ax6_twin.tick_params(colors=NEU, labelsize=7)
        ax6_twin.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax6_twin.set_ylabel("Drawdown", fontsize=8, color=NEU)

    ax6.set_title("Stitched OOS Equity Curve", fontsize=11, fontweight="bold", color="white", pad=8)
    ax6.set_ylabel("Cumulative Return (×1)", fontsize=9, color=NEU)
    ax6.set_facecolor(CARD)
    for sp in ax6.spines.values(): sp.set_edgecolor("#2a2a2a")
    ax6.grid(True, alpha=0.15, color=GRID)
    ax6.legend(fontsize=8, loc="upper left", facecolor="#1a1a1a", edgecolor="#333")
    ax6.tick_params(colors=NEU, labelsize=8)
    ax6.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # ── Metrics table in figure margin ───────────────────────────────────────
    v5_avg_sr  = v5["oos_sharpe"].mean()
    v5_avg_srt = v5["oos_sortino"].mean()
    v5_avg_mdd = v5["oos_max_dd"].mean()
    v5_avg_cagr= v5["oos_cagr"].mean()
    v5_avg_turn= v5["avg_turnover"].mean()

    b_avg_sr   = base["oos_sharpe"].mean()   if base is not None and "oos_sharpe"  in base else float("nan")
    b_avg_srt  = base["oos_sortino"].mean()  if base is not None and "oos_sortino" in base else float("nan")
    b_avg_mdd  = base["oos_max_dd"].mean()   if base is not None and "oos_max_dd"  in base else float("nan")

    summary_text = (
        f"v5 avg Sharpe  : {v5_avg_sr:+.3f}   (Δ {v5_avg_sr - b_avg_sr:+.3f})\n"
        f"v5 avg Sortino : {v5_avg_srt:+.3f}   (Δ {v5_avg_srt - b_avg_srt:+.3f})\n"
        f"v5 avg MaxDD   : {v5_avg_mdd:+.2%}  (Δ {v5_avg_mdd - b_avg_mdd:+.2%})\n"
        f"v5 avg CAGR    : {v5_avg_cagr:+.2%}\n"
        f"v5 avg Turnover: {v5_avg_turn:.2%}"
    )
    fig.text(
        0.02, 0.005, summary_text,
        fontsize=8.5, color="#aaaaaa",
        fontfamily="monospace",
        va="bottom",
    )

    fig.savefig(_PLOT_OUT, dpi=160, bbox_inches="tight", facecolor=BG)
    logger.info("✅ Chart saved → %s", _PLOT_OUT)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()