"""
FORTRESS v5 - signals/options_alpha.py  [v9 — VRP TIME-SERIES + FORCED VTS PROXY]

BUG FIX (v9.1):
  VTS was 0.0% active because VIX is stored in decimal form (0.15-0.35)
  after the /100.0 conversion in load_data(), but _compute_vts_zscores_full()
  called clip(lower=1.0) on it. Every value (0.15-0.35) clamped to 1.0
  → constant series → pct_change=0 → delta_z=0 → VTS dead for entire run.

  FIX: clip(lower=0.01) — allows decimal VIX to pass through unchanged.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("OptionsAlpha")

_CBOE_ETF_VOL_MAP: Dict[str, str] = {
    "SPY": "^VIX",
    "QQQ": "^VXN",
    "GLD": "^GVZ",
    "USO": "^OVX",
}

_UNIVERSE: List[str] = [
    "SPY", "QQQ", "IWM", "TLT", "HYG", "LQD", "GLD", "SLV",
    "GDX", "XLE", "XLF", "XLK", "XLV", "XLU", "XLI", "XLP",
    "XLY", "XLB", "XLC", "VIXY", "BIL", "SHV", "USO", "PDBC", "COWZ",
]
_CASH_EQUIV_TICKERS: frozenset[str] = frozenset({"BIL", "SHV"})

_RV_WINDOW:       int = 21
_VRP_TS_HALFLIFE: int = 126
_VRP_SMOOTH_HL:   int = 5
_VTS_DELTA_DAYS:  int = 5
_VTS_PROXY_HL:    int = 10


class OptionsAlphaEngine:
    """
    VRP (per-asset time-series z-score, CBOE tickers only) and
    VTS (VIX 5-day momentum proxy, always active) signals.
    """

    def __init__(self, rv_window: int = _RV_WINDOW) -> None:
        self._rv_w:     int                      = rv_window
        self._prices:   Optional[pd.DataFrame]   = None
        self._returns:  Optional[pd.DataFrame]   = None
        self._cboe_iv:  Optional[pd.DataFrame]   = None
        self._vts:      Optional[pd.DataFrame]   = None
        self._vts_9d_available: bool             = False
        self._iv_matrix:  Optional[pd.DataFrame] = None
        self._rv_matrix:  Optional[pd.DataFrame] = None
        self._vrp_matrix: Optional[pd.DataFrame] = None
        self._vts_matrix: Optional[pd.DataFrame] = None

    async def load_data(self, start: str = "2015-01-01") -> None:
        logger.info(f"  Fetching prices for {len(_UNIVERSE)} tickers from {start}...")
        raw = yf.download(_UNIVERSE, start=start, auto_adjust=True, progress=False)
        if raw.empty:
            raise RuntimeError("yfinance returned empty DataFrame for universe.")

        prices = (
            raw["Close"] if "Close" in raw.columns
            else raw.xs("Close", axis=1, level=0)
        )
        self._prices  = prices.reindex(columns=_UNIVERSE).ffill()
        self._returns = self._prices.pct_change().fillna(0.0)

        # Fetch CBOE vol indices
        cboe_tickers = ["^VIX", "^VXN", "^GVZ", "^OVX"]
        try:
            cboe_raw_all = yf.download(
                cboe_tickers, start=start, auto_adjust=True, progress=False
            )
            if not cboe_raw_all.empty:
                cboe_raw = (
                    cboe_raw_all["Close"] if "Close" in cboe_raw_all.columns
                    else cboe_raw_all.xs("Close", axis=1, level=0)
                ) / 100.0   # percent → decimal annualised vol (e.g. 0.15 = 15%)

                cboe_iv_raw = pd.DataFrame(index=cboe_raw.index)
                for etf, vol_ticker in _CBOE_ETF_VOL_MAP.items():
                    if vol_ticker in cboe_raw.columns and cboe_raw[vol_ticker].notna().sum() > 10:
                        cboe_iv_raw[etf] = cboe_raw[vol_ticker]
                    else:
                        logger.info(f"CBOE {vol_ticker} unavailable — {etf} uses RV fallback")

                self._cboe_iv = cboe_iv_raw.reindex(self._prices.index).ffill()

                # VTS always uses proxy (VIX9D too unreliable pre-2022)
                # VIX here is in decimal form (0.15-0.35) — NOT raw (15-35)
                vix_col = cboe_raw.get("^VIX", pd.Series(dtype=float))
                self._vts = pd.DataFrame({"VIX": vix_col}).reindex(
                    self._prices.index
                ).ffill()
                self._vts_9d_available = False
                logger.info("  VTS mode: vix_proxy (forced — VIX9D unreliable pre-2022)")

            else:
                logger.warning("CBOE vol indices unavailable — using RV for all tickers")
                self._cboe_iv = pd.DataFrame(index=self._prices.index)
                self._vts     = pd.DataFrame(index=self._prices.index)

        except Exception as e:
            logger.warning(f"CBOE download failed ({e}) — RV fallback for all")
            self._cboe_iv = pd.DataFrame(index=self._prices.index)
            self._vts     = pd.DataFrame(index=self._prices.index)

        self._precompute_all()

        cboe_cov   = len(self._cboe_iv.columns) if self._cboe_iv is not None else 0
        vrp_active = (self._vrp_matrix.abs() > 0.01).any(axis=1).mean() * 100
        vts_active = (self._vts_matrix.abs() > 0.01).any(axis=1).mean() * 100
        logger.info(
            f"OptionsAlpha loaded: {len(self._prices)} days x {len(_UNIVERSE)} assets | "
            f"CBOE IV coverage: {cboe_cov} tickers | "
            f"VRP active: {vrp_active:.1f}% | VTS active: {vts_active:.1f}%"
        )

    def _precompute_all(self) -> None:
        assert self._prices is not None and self._returns is not None
        logger.info("  Precomputing IV, RV, VRP, VTS matrices...")
        self._iv_matrix  = self._compute_iv_matrix_full()
        self._rv_matrix  = self._compute_rv_matrix_full()
        self._vrp_matrix = self._compute_vrp_zscores_full()
        self._vts_matrix = self._compute_vts_zscores_full()
        vrp_active = (self._vrp_matrix.abs() > 0.01).any(axis=1).mean() * 100
        vts_active = (self._vts_matrix.abs() > 0.01).any(axis=1).mean() * 100
        logger.info(
            f"  Precompute complete | VRP active: {vrp_active:.1f}% | "
            f"VTS active: {vts_active:.1f}%"
        )

    def _compute_iv_matrix_full(self) -> pd.DataFrame:
        """(T, N) annualised IV. CBOE settlement-grade for covered tickers, EWMA RV otherwise."""
        rv_ewm = (
            self._returns
            .reindex(columns=_UNIVERSE)
            .ewm(span=self._rv_w * 2, min_periods=self._rv_w)
            .std()
            * np.sqrt(252)
        )
        iv_matrix = rv_ewm.clip(lower=0.02)

        if self._cboe_iv is not None and not self._cboe_iv.empty:
            for etf in _CBOE_ETF_VOL_MAP:
                if etf in self._cboe_iv.columns and etf in iv_matrix.columns:
                    cboe_series = self._cboe_iv[etf].reindex(iv_matrix.index).ffill()
                    valid_mask  = cboe_series.notna() & (cboe_series > 0.01)
                    iv_matrix.loc[valid_mask, etf] = cboe_series[valid_mask]

        for ticker in _CASH_EQUIV_TICKERS:
            if ticker in iv_matrix.columns:
                iv_matrix[ticker] = 0.02

        return iv_matrix.ffill().fillna(0.15)

    def _compute_rv_matrix_full(self) -> pd.DataFrame:
        """(T, N) 21d rolling realized vol, annualised."""
        rv = (
            self._returns
            .reindex(columns=_UNIVERSE)
            .rolling(self._rv_w, min_periods=max(self._rv_w // 2, 5))
            .std()
            * np.sqrt(252)
        )
        for ticker in _CASH_EQUIV_TICKERS:
            if ticker in rv.columns:
                rv[ticker] = 0.02
        return rv.clip(lower=0.02).fillna(0.15)

    def _compute_vrp_zscores_full(self) -> pd.DataFrame:
        """
        (T, N) VRP time-series EWMA z-score for CBOE tickers only.

        VRP_i(t)  = IV_i(t) - RV_i(t)
        z_ts_i(t) = (VRP_i - EWMA_mean_i) / EWMA_std_i   [halflife=126d]
        signal_i  = tanh(+z_ts_i * 0.75)

        Non-CBOE tickers: 0.0 (IV ~= RV, no VRP signal content).
        CAUSAL: EWMA uses only data through t.
        """
        assert self._iv_matrix is not None and self._rv_matrix is not None

        vrp_raw = self._iv_matrix - self._rv_matrix
        result  = pd.DataFrame(0.0, index=vrp_raw.index, columns=_UNIVERSE)

        for ticker in _CBOE_ETF_VOL_MAP:
            if ticker not in vrp_raw.columns:
                continue
            vrp_series = vrp_raw[ticker]
            ewm        = vrp_series.ewm(halflife=_VRP_TS_HALFLIFE, min_periods=21)
            z_ts       = (vrp_series - ewm.mean()) / ewm.std().clip(lower=1e-4)
            result[ticker] = np.tanh(z_ts.clip(-3.0, 3.0) * 0.75)

        for ticker in _CBOE_ETF_VOL_MAP:
            if ticker in result.columns:
                result[ticker] = (
                    result[ticker].ewm(halflife=_VRP_SMOOTH_HL, min_periods=2).mean()
                )

        return result

    def _compute_vts_zscores_full(self) -> pd.DataFrame:
        """
        (T, N) VTS signal — VIX 5-day momentum proxy.

        PROXY: EWMA_z(-VIX.pct_change(5), halflife=10)
          VIX falling 5d → term structure improving → bullish (+)
          VIX rising  5d → term structure worsening → bearish (-)

        VIX IS IN DECIMAL FORM (0.15-0.35) after /100.0 in load_data().
        clip(lower=0.01) — NOT clip(lower=1.0) which would clamp all values to 1.0.

        Routing: equity_weight per asset. Bonds/commodities receive 0.
        """
        if self._vts is None or self._vts.empty or "VIX" not in self._vts.columns:
            logger.warning("  VTS: no VIX data — returning zero matrix")
            return pd.DataFrame(0.0, index=self._prices.index, columns=_UNIVERSE)

        # BUG FIX: VIX stored as decimal (0.15), was incorrectly clipped at 1.0
        vix       = self._vts["VIX"].clip(lower=0.01)   # <-- was clip(lower=1.0), BUG FIXED
        vix_mom   = -vix.pct_change(_VTS_DELTA_DAYS)
        proxy_ewm = vix_mom.ewm(halflife=_VTS_PROXY_HL, min_periods=5)
        delta_z   = (
            (vix_mom - proxy_ewm.mean()) / proxy_ewm.std().clip(lower=1e-6)
        ).clip(-3, 3)

        equity_weights = pd.Series({t: _get_equity_weight(t) for t in _UNIVERSE})

        vts_matrix = pd.DataFrame(
            np.outer(delta_z.values, equity_weights.values),
            index=delta_z.index,
            columns=_UNIVERSE,
        ).reindex(self._prices.index).ffill().fillna(0.0)

        return np.tanh(vts_matrix * 0.6)

    def compute_vrp_history(self) -> pd.DataFrame:
        """(T, N) VRP z-score history. Non-zero only for SPY, QQQ, GLD, USO."""
        if self._vrp_matrix is None:
            raise RuntimeError("Call await load_data() first.")
        return self._vrp_matrix.copy()

    def compute_vts_history(self) -> pd.DataFrame:
        """(T, N) VTS history. Always VIX proxy. Should be ~85-90% active after fix."""
        if self._vts_matrix is None:
            raise RuntimeError("Call await load_data() first.")
        return self._vts_matrix.copy()

    def get_signal_summary(self) -> dict:
        if self._vrp_matrix is None:
            return {"error": "Not loaded"}
        vrp_active = (self._vrp_matrix.abs() > 0.01).any(axis=1).mean()
        vts_active = (self._vts_matrix.abs() > 0.01).any(axis=1).mean()
        vrp_ticker_means = {
            t: round(float(self._vrp_matrix[t].mean()), 4)
            for t in list(_CBOE_ETF_VOL_MAP.keys()) + ["IWM", "TLT", "BIL"]
            if t in self._vrp_matrix.columns
        }
        vrp_ticker_stds = {
            t: round(float(self._vrp_matrix[t].std()), 4)
            for t in _CBOE_ETF_VOL_MAP if t in self._vrp_matrix.columns
        }
        return {
            "vrp_mean_abs":     float(self._vrp_matrix.abs().mean().mean()),
            "vts_mean_abs":     float(self._vts_matrix.abs().mean().mean()),
            "vrp_active_pct":   float(vrp_active * 100),
            "vts_active_pct":   float(vts_active * 100),
            "vts_mode":         "vix_proxy (forced)",
            "vrp_ticker_means": vrp_ticker_means,
            "vrp_ticker_stds":  vrp_ticker_stds,
        }


def _get_equity_weight(ticker: str) -> float:
    pure_equity  = {
        "SPY", "QQQ", "IWM", "XLK", "XLF", "XLV", "XLU",
        "XLI", "XLP", "XLY", "XLB", "XLC", "VIXY", "COWZ",
    }
    mixed_equity = {"GDX": 0.6, "XLE": 0.7, "PDBC": 0.3}
    if ticker in pure_equity:
        return 1.0
    if ticker in mixed_equity:
        return mixed_equity[ticker]
    if ticker in {"TLT", "HYG", "LQD", "BIL", "SHV", "GLD", "SLV", "USO"}:
        return 0.0
    return 0.5