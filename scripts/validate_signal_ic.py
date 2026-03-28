"""
FORTRESS v5 - scripts/validate_signal_ic.py  [v2.1 — MIN_ASSETS BUG FIX]

BUG FIX (v2.1):
  _MIN_ASSETS was 5, but VRP only fires on 4 CBOE tickers (SPY, QQQ, GLD, USO).
  Every date had exactly 4 active assets → below threshold → INSUFFICIENT_DATA.
  The VRP signal had genuine IC_63d=0.045 but the validator couldn't see it.

  FIX: _MIN_ASSETS = 3. This allows VRP (4 tickers) and any other sparse signals
  to be properly evaluated. Using 3 as the floor because Spearman correlation
  is only meaningful with at least 3 ranked pairs.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("SignalIC")

_CACHE     = Path("research/outputs/cache")
_RETURNS   = _CACHE / "returns_wide.parquet"
_SIGNALS   = _CACHE / "alpha_signals.parquet"
_REGIME    = _CACHE / "regime_posteriors.parquet"

_HORIZONS:       List[int]  = [1, 5, 21, 63]
_IC_MIN_VIABLE:  float      = 0.03
_IC_STRONG:      float      = 0.05
_ICIR_MIN:       float      = 0.50
_ICIR_STRONG:    float      = 1.00
_MIN_DATES:      int        = 63
_MIN_ASSETS:     int        = 3     # BUG FIX: was 5, VRP only has 4 CBOE tickers
_ACTIVE_THRESH:  float      = 0.005


def _compute_forward_returns(returns_arr: np.ndarray, horizon: int) -> np.ndarray:
    T, N  = returns_arr.shape
    fwd   = np.full((T, N), np.nan)
    for t in range(T - horizon):
        fwd[t] = np.prod(1.0 + returns_arr[t+1:t+horizon+1], axis=0) - 1.0
    return fwd


def _compute_ic_series(
    signal_arr:    np.ndarray,
    fwd_ret_arr:   np.ndarray,
    active_thresh: float = _ACTIVE_THRESH,
    min_assets:    int   = _MIN_ASSETS,
) -> np.ndarray:
    """
    Per-date cross-sectional Spearman IC, computed only on active dates.

    Active = at least min_assets assets have |signal| > active_thresh.
    Inactive dates → NaN. This means the IC measures only the signal's
    claim on dates it actually fires — not the vacuous zero-signal days.
    """
    T, N      = signal_arr.shape
    ic_series = np.full(T, np.nan)

    for t in range(T):
        s      = signal_arr[t]
        r      = fwd_ret_arr[t]
        active = (np.abs(s) > active_thresh) & ~np.isnan(r)
        if active.sum() < min_assets:
            continue
        ic_val, _ = scipy_stats.spearmanr(s[active], r[active])
        if np.isfinite(ic_val):
            ic_series[t] = ic_val

    return ic_series


def _ic_mean_and_icir(
    ic_series: np.ndarray,
    min_dates: int = _MIN_DATES,
) -> Tuple[float, float]:
    valid = ic_series[~np.isnan(ic_series)]
    if len(valid) < min_dates:
        return np.nan, np.nan
    ic_mean = float(np.mean(valid))
    ic_std  = float(np.std(valid))
    icir    = ic_mean / (ic_std + 1e-10) if ic_std > 1e-10 else np.nan
    return ic_mean, icir


def _compute_ic_full(
    signal_arr:  np.ndarray,
    returns_arr: np.ndarray,
    horizons:    List[int],
) -> Dict[int, float]:
    results = {}
    T = signal_arr.shape[0]
    for k in horizons:
        if k >= T - 1:
            results[k] = np.nan
            continue
        fwd        = _compute_forward_returns(returns_arr, k)
        ic_series  = _compute_ic_series(signal_arr, fwd)
        ic_mean, _ = _ic_mean_and_icir(ic_series)
        results[k] = ic_mean
    return results


def _compute_ic_by_regime(
    signal_arr:  np.ndarray,
    returns_arr: np.ndarray,
    regime_df:   pd.DataFrame,
    dates:       pd.DatetimeIndex,
    horizon:     int = 5,
) -> Dict[str, float]:
    if "equity_label" not in regime_df.columns:
        return {}
    labels = regime_df["equity_label"].reindex(dates).ffill().fillna("unknown")
    fwd    = _compute_forward_returns(returns_arr, horizon)
    results = {}
    for label in ["crisis", "stress", "neutral", "complacent"]:
        mask = (labels.values == label)
        if mask.sum() < _MIN_DATES:
            results[label] = np.nan
            continue
        ic_series    = _compute_ic_series(signal_arr[mask], fwd[mask])
        ic_mean, _   = _ic_mean_and_icir(ic_series, min_dates=20)
        results[label] = ic_mean
    return results


def _verdict(ic_5d: float, icir: float) -> Tuple[str, str]:
    if np.isnan(ic_5d):
        return "INSUFFICIENT_DATA", "not enough active dates"
    if abs(ic_5d) < 0.005:
        return "NO_SIGNAL", "IC indistinguishable from zero — abandon signal"
    if ic_5d < -0.005:
        return "REVERSE_SIGNAL", f"IC={ic_5d:.4f} — try negating the signal"
    if ic_5d < _IC_MIN_VIABLE:
        stability = f"ICIR={icir:.2f}" if not np.isnan(icir) else "ICIR=insufficient data"
        return "WEAK", f"IC={ic_5d:.4f} below viable threshold ({_IC_MIN_VIABLE}) | {stability}"
    if ic_5d >= _IC_STRONG and not np.isnan(icir) and icir >= _ICIR_MIN:
        return "STRONG", f"IC={ic_5d:.4f} ICIR={icir:.2f} — pursue aggressively"
    if ic_5d >= _IC_MIN_VIABLE:
        stab = (
            f"stable ICIR={icir:.2f}" if (not np.isnan(icir) and icir >= _ICIR_MIN) else
            f"unstable ICIR={icir:.2f}" if not np.isnan(icir) else "ICIR needs more data"
        )
        return "VIABLE", f"IC={ic_5d:.4f} {stab}"
    return "BORDERLINE", f"IC={ic_5d:.4f}"


def validate_signal(
    signal_name: str,
    signal_arr:  np.ndarray,
    returns_arr: np.ndarray,
    dates:       pd.DatetimeIndex,
    regime_df:   Optional[pd.DataFrame] = None,
    tickers:     Optional[List[str]]    = None,
) -> Tuple[float, float]:
    T, N           = signal_arr.shape
    active_pct     = float((np.abs(signal_arr) > _ACTIVE_THRESH).any(axis=1).mean()) * 100
    n_active_dates = int((np.abs(signal_arr) > _ACTIVE_THRESH).any(axis=1).sum())

    logger.info(f"\n{'='*60}")
    logger.info(f"  Signal: {signal_name}")
    logger.info(f"  Shape: ({T}, {N}) | Active: {active_pct:.1f}% ({n_active_dates} dates)")
    logger.info(f"  Mean |signal|: {np.abs(signal_arr).mean():.4f} | "
                f"Max |signal|: {np.abs(signal_arr).max():.4f}")

    ic_by_horizon = _compute_ic_full(signal_arr, returns_arr, _HORIZONS)
    logger.info("  IC by forward horizon (active dates only):")
    for k, ic in ic_by_horizon.items():
        if np.isnan(ic):
            logger.info(f"    {k:>3}d:    nan  (insufficient active dates)")
        else:
            bar  = "+" * max(0, int(ic * 100)) if ic > 0 else "-" * max(0, int(abs(ic) * 100))
            flag = " ✓" if ic >= _IC_MIN_VIABLE else (" ~" if ic >= 0.01 else " ✗")
            logger.info(f"    {k:>3}d: {ic:+.4f} {bar}{flag}")

    fwd5       = _compute_forward_returns(returns_arr, 5)
    ic5_series = _compute_ic_series(signal_arr, fwd5)
    ic5_mean, icir5 = _ic_mean_and_icir(ic5_series)
    n_ic5_valid     = int(np.isfinite(ic5_series).sum())

    if np.isnan(icir5):
        logger.info(f"  ICIR (5d): nan  ({n_ic5_valid} active dates < {_MIN_DATES} required)")
    else:
        stab = "stable" if icir5 >= _ICIR_MIN else "unstable"
        logger.info(
            f"  ICIR (5d): {icir5:.3f} ({stab}) | from {n_ic5_valid} active dates"
        )

    if regime_df is not None:
        ic_regime = _compute_ic_by_regime(signal_arr, returns_arr, regime_df, dates)
        if ic_regime:
            logger.info("  IC by regime (5d, active dates only):")
            for label, ic in ic_regime.items():
                if np.isnan(ic):
                    logger.info(f"    {label:>12s}:    nan  (insufficient)")
                else:
                    flag = " ✓" if ic >= _IC_MIN_VIABLE else ""
                    logger.info(f"    {label:>12s}: {ic:+.4f}{flag}")

    primary_ic    = ic_by_horizon.get(5, np.nan)
    verdict, expl = _verdict(primary_ic, icir5)
    symbols = {
        "STRONG":"✓✓", "VIABLE":"✓", "BORDERLINE":"~", "WEAK":"✗",
        "NO_SIGNAL":"✗✗", "REVERSE_SIGNAL":"↕", "INSUFFICIENT_DATA":"?",
    }
    logger.info(f"\n  VERDICT [{symbols.get(verdict,'?')}] {verdict}: {expl}")
    logger.info(f"{'='*60}")

    return (float(primary_ic) if not np.isnan(primary_ic) else np.nan, icir5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast signal IC validation (v2.1 — MIN_ASSETS=3 for sparse signals)"
    )
    parser.add_argument("--signal",      type=str, default=None)
    parser.add_argument("--signal-file", type=str, default=None)
    parser.add_argument("--horizon",     type=int, default=None)
    args = parser.parse_args()

    if not _RETURNS.exists():
        logger.error(f"Returns file not found: {_RETURNS}")
        sys.exit(1)

    returns_df = pd.read_parquet(_RETURNS)
    returns_df.index = pd.to_datetime(returns_df.index)
    returns_df.sort_index(inplace=True)

    TICKERS     = list(returns_df.columns)
    dates       = returns_df.index
    returns_arr = returns_df.values.astype(np.float64)

    regime_df: Optional[pd.DataFrame] = None
    if _REGIME.exists():
        regime_df = pd.read_parquet(_REGIME)
        regime_df.index = pd.to_datetime(regime_df.index)

    if args.signal_file:
        signal_path = Path(args.signal_file)
        if not signal_path.exists():
            logger.error(f"Signal file not found: {signal_path}")
            sys.exit(1)
        custom_df = pd.read_parquet(signal_path)
        custom_df.index = pd.to_datetime(custom_df.index)
        custom_df = (
            custom_df.reindex(dates).ffill().fillna(0.0)
            .reindex(columns=TICKERS).fillna(0.0)
        )
        validate_signal(
            signal_name=signal_path.stem,
            signal_arr=custom_df.values.astype(np.float64),
            returns_arr=returns_arr,
            dates=dates,
            regime_df=regime_df,
            tickers=TICKERS,
        )
        return

    if not _SIGNALS.exists():
        logger.error(
            f"Signal matrix not found: {_SIGNALS}. "
            "Run precompute_alpha_signals.py first."
        )
        sys.exit(1)

    signals_df = pd.read_parquet(_SIGNALS)
    signals_df.index = pd.to_datetime(signals_df.index)
    signals_df = signals_df.reindex(dates).ffill().fillna(0.0)

    all_signal_names = sorted(set(c.rsplit("_", 1)[0] for c in signals_df.columns))

    if args.signal:
        if args.signal not in all_signal_names:
            logger.error(f"Signal '{args.signal}' not found. Available: {all_signal_names}")
            sys.exit(1)
        all_signal_names = [args.signal]

    logger.info(
        f"IC validation (v2.1) for {len(all_signal_names)} signals | "
        f"{len(dates)} dates | {len(TICKERS)} assets"
    )
    logger.info(
        f"Thresholds: IC_viable={_IC_MIN_VIABLE} | IC_strong={_IC_STRONG} | "
        f"ICIR_min={_ICIR_MIN} | min_active_dates={_MIN_DATES} | min_assets={_MIN_ASSETS}"
    )

    summary_rows = []

    for sig_name in all_signal_names:
        cols = [c for c in signals_df.columns if c.startswith(f"{sig_name}_")]
        if not cols:
            continue
        sig_df = signals_df[cols].copy()
        sig_df.columns = [c.replace(f"{sig_name}_", "") for c in cols]
        sig_df = sig_df.reindex(columns=TICKERS).fillna(0.0)

        common = sig_df.index.intersection(returns_df.index)
        s_arr  = sig_df.reindex(common).values.astype(np.float64)
        r_arr  = returns_df.reindex(common).values.astype(np.float64)
        d_arr  = common

        ic5, icir5 = validate_signal(
            signal_name=sig_name,
            signal_arr=s_arr,
            returns_arr=r_arr,
            dates=d_arr,
            regime_df=regime_df,
            tickers=TICKERS,
        )

        ic_21d = _compute_ic_full(s_arr, r_arr, [21])[21]
        active = float((np.abs(s_arr) > _ACTIVE_THRESH).any(axis=1).mean()) * 100
        v, _   = _verdict(ic5, icir5)
        summary_rows.append({
            "signal":  sig_name,
            "IC_5d":   ic5,
            "IC_21d":  ic_21d,
            "ICIR_5d": icir5,
            "active%": active,
            "verdict": v,
        })

    if len(summary_rows) > 1:
        logger.info(f"\n{'='*60}")
        logger.info("  SUMMARY (ranked by IC_5d)")
        logger.info(f"{'='*60}")
        summary = pd.DataFrame(summary_rows).sort_values("IC_5d", ascending=False)
        for _, row in summary.iterrows():
            ic5_s  = f"{row['IC_5d']:+.4f}" if not np.isnan(row['IC_5d']) else "   nan"
            ic21_s = f"{row['IC_21d']:+.4f}" if not np.isnan(row['IC_21d']) else "   nan"
            icir_s = f"{row['ICIR_5d']:+.2f}" if not np.isnan(row['ICIR_5d']) else "  nan"
            flag   = "✓" if (not np.isnan(row['IC_5d']) and row['IC_5d'] >= _IC_MIN_VIABLE) else "✗"
            logger.info(
                f"  {flag} {row['signal']:12s}  IC_5d={ic5_s}  IC_21d={ic21_s}  "
                f"ICIR={icir_s}  active={row['active%']:.0f}%  [{row['verdict']}]"
            )

    logger.info(
        "\nNext steps:\n"
        "  VIABLE/STRONG  -> keep, weight by IC\n"
        "  WEAK IC_21d>0  -> keep with lambda_turn=0.05 (holds 20-30d)\n"
        "  REVERSE_SIGNAL -> flip sign, re-validate\n"
        "  NO_SIGNAL      -> drop, redistribute weight\n"
    )


if __name__ == "__main__":
    main()