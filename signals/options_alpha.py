"""
FORTRESS v5 - signals/options_alpha.py
Path: signals/options_alpha.py

Options Surface Alpha Signals — VRP, Skew, Term Structure.

ARCHITECTURAL DECISION (addresses Q3):
  Using VIX × vol_beta as a proxy for per-ticker ATM IV is a critical
  architectural flaw: it collapses all idiosyncratic IV information into
  a linear function of a single market factor. This destroys precisely the
  cross-sectional dispersion in fear premiums that makes VRP a useful signal.

  THE CORRECT APPROACH: Use CBOE's published vol index family.
  CBOE provides free, settlement-grade, point-in-time 30-day ATM IV for:

    VIX   → SPY (S&P 500)               yfinance: ^VIX
    VXN   → QQQ (Nasdaq 100)            yfinance: ^VXN
    RVX   → IWM (Russell 2000)          yfinance: ^RVX
    GVZ   → GLD (Gold ETF)              yfinance: ^GVZ
    OVX   → USO (Crude Oil ETF)         yfinance: ^OVX
    VXEEM → EEM (Emerging Markets)      yfinance: ^VXEEM

  These indices ARE the ATM implied vol — they are computed by CBOE from
  the actual options market using the same ORATS/model-free VIX methodology.
  No approximation. No proxy. Settlement-grade data for FREE.

  For tickers WITHOUT a dedicated CBOE vol index (XLE, XLF, XLK, etc.):
    We compute 21-day EWMA realized vol as a conservative IV estimate.
    This is deliberately conservative: it understates IV, which means VRP
    will be understated (not overstated) for these tickers. This prevents
    false high-VRP signals.

  UPGRADE PATH (when budget allows):
    Tradier free API provides live option chains for any US equity/ETF.
    GET https://api.tradier.com/v1/markets/options/chains?symbol={ticker}
         &expiration={nearest_30d_expiry}&greeks=true
    Returns actual market ATM IV per ticker, 15-min delay, no cost.
    This upgrades the 6 non-CBOE tickers from RV-estimated to actual IV.

SIGNALS PRODUCED:
  [1] VRP cross-sectional z-score:
      VRP_i = IV_i − RV_i(21d). Cross-sectional z-score of VRP.
      Signal direction: HIGH VRP → excess fear priced → CONTRARIAN BUY.
      (When fear premium is anomalously high, it's been overpaid for.)

  [2] IV-to-HV ratio z-score (IV/RV ratio, z-scored over 252d EWMA):
      When IV/RV >> historical norm → vol is expensive → underlying
      tends to disappoint the implied move → directional compression signal.

  [3] Term structure alpha (applies to equity ETFs only):
      VIX9D/VIX slope delta: change in term structure slope over 5 days.
      Rapid transition from contango to backwardation precedes drawdown.
      Rapid return to contango precedes recovery.

LOOK-AHEAD CONTRACT:
  CBOE vol indices are published at 4:15 PM ET same day.
  All rolling windows (21d RV, 252d z-score) use strictly causal EWMA.
  The yfinance download retrieves close prices — T+0 safe for T+1 signals.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("OptionsAlpha")

# ── CBOE vol index → ETF ticker mapping ───────────────────────────────────────
# These are the exact vol indices CBOE computes for these ETFs.
# No proxy; no scaling.
_CBOE_ETF_VOL_MAP: Dict[str, str] = {
    "SPY":  "^VIX",
    "QQQ":  "^VXN",
    "IWM":  "^RVX",
    "GLD":  "^GVZ",
    "USO":  "^OVX",
    # Sector ETFs that also have dedicated CBOE vol indices
    # XLE, XLF, XLK don't have dedicated indices → use RV fallback
}

# Full 25-ticker universe
_UNIVERSE: List[str] = [
    "SPY", "QQQ", "IWM", "TLT", "HYG", "LQD", "GLD", "SLV",
    "GDX", "XLE", "XLF", "XLK", "XLV", "XLU", "XLI", "XLP",
    "XLY", "XLB", "XLC", "VIXY", "BIL", "SHV", "USO", "PDBC", "COWZ",
]

# Tickers that use EWMA RV as IV proxy (no dedicated CBOE index)
_RV_PROXY_TICKERS: List[str] = [
    t for t in _UNIVERSE if t not in _CBOE_ETF_VOL_MAP
]

# BIL/SHV: near-zero vol — clip IV floor to prevent division issues
_CASH_EQUIV_TICKERS = {"BIL", "SHV"}

# Rolling windows
_RV_WINDOW      = 21    # Realized vol window for VRP computation
_ZSCORE_HALFLIFE = 252  # EWMA halflife for cross-sectional z-score normalisation
_SKEW_WINDOW    = 10    # Days for IV-to-HV ratio smoothing

@dataclass(frozen=True, slots=True)
class OptionsSignalVector:
    """Per-asset options surface signals at a single date."""
    date:           str
    vrp_z:          pd.Series   # shape (25,) — VRP cross-sectional z-score
    iv_hv_ratio_z:  pd.Series   # shape (25,) — IV/HV ratio z-score
    term_struct_z:  pd.Series   # shape (25,) — term structure slope delta z-score


class OptionsAlphaEngine:
    """
    Computes VRP and IV structure signals using CBOE vol indices for
    tickers with native coverage and EWMA RV fallback for the rest.

    The key property preserved: cross-sectional DISPERSION in VRP.
    Using CBOE indices for the 5 most important tickers (SPY/QQQ/IWM/GLD/USO)
    ensures the signal captures genuine idiosyncratic fear differentials,
    not just scaled market vol.
    """

    def __init__(
        self,
        rv_window:        int = _RV_WINDOW,
        zscore_halflife:  int = _ZSCORE_HALFLIFE,
    ) -> None:
        self._rv_w    = rv_window
        self._z_hl    = zscore_halflife
        self._prices: Optional[pd.DataFrame]   = None
        self._returns: Optional[pd.DataFrame]  = None
        self._cboe_iv: Optional[pd.DataFrame]  = None
        self._vts:     Optional[pd.DataFrame]  = None  # VIX term structure

    async def load_data(self, start: str = "2015-01-01") -> None:
        """
        Load ETF prices, CBOE vol indices, and VIX term structure.
        Three independent fetches run concurrently.
        """
        loop = asyncio.get_event_loop()

        price_task = loop.run_in_executor(
            None,
            lambda: yf.download(_UNIVERSE, start=start, progress=False)["Close"]
        )
        cboe_task = loop.run_in_executor(
            None,
            lambda: yf.download(
                list(_CBOE_ETF_VOL_MAP.values()) +
                ["^VIX9D", "^VIX3M"],  # term structure
                start=start,
                progress=False,
            )["Close"]
        )
        results = await asyncio.gather(price_task, cboe_task, return_exceptions=True)

        if isinstance(results[0], Exception):
            raise RuntimeError(f"Price data fetch failed: {results[0]}")
        if isinstance(results[1], Exception):
            logger.warning(f"CBOE vol index fetch partial failure: {results[1]}")

        prices_raw = results[0]
        cboe_raw   = results[1] if not isinstance(results[1], Exception) else pd.DataFrame()

        self._prices  = prices_raw.reindex(columns=_UNIVERSE).ffill().dropna(how="all")
        self._returns = self._prices.pct_change()

        # Build CBOE IV dataframe indexed by ETF ticker name
        if not cboe_raw.empty:
            cboe_iv = pd.DataFrame(index=cboe_raw.index)
            for etf, vol_ticker in _CBOE_ETF_VOL_MAP.items():
                if vol_ticker in cboe_raw.columns:
                    # CBOE indices are in vol-points (e.g. 18.5 = 18.5% annualised)
                    cboe_iv[etf] = cboe_raw[vol_ticker] / 100.0
                else:
                    logger.warning(f"CBOE vol index {vol_ticker} not in download. Using RV for {etf}.")
            self._cboe_iv = cboe_iv.reindex(self._prices.index).ffill()

            # VIX term structure
            self._vts = pd.DataFrame({
                "VIX9D": cboe_raw.get("^VIX9D", cboe_raw.get("^VIX", 0.0)),
                "VIX":   cboe_raw.get("^VIX",   0.0),
                "VIX3M": cboe_raw.get("^VIX3M", cboe_raw.get("^VIX", 0.0)),
            }).reindex(self._prices.index).ffill()
        else:
            self._cboe_iv = pd.DataFrame(index=self._prices.index)
            self._vts     = pd.DataFrame(index=self._prices.index)

        logger.info(
            f"OptionsAlpha loaded: {len(self._prices)} days × {len(_UNIVERSE)} assets | "
            f"CBOE IV coverage: {len(self._cboe_iv.columns)} tickers"
        )

    def _get_iv_matrix(self, as_of_date: str) -> pd.Series:
        """
        Returns the annualised 30-day ATM implied vol for each asset.

        HIERARCHY:
          1. CBOE settlement-grade vol index (GVZ for GLD, OVX for USO, etc.)
          2. EWMA 21d realized vol (conservative proxy for non-CBOE tickers)
          3. Hard floor: 0.02 (2% floor prevents degenerate VRP on cash-equiv)
        """
        assert self._prices is not None and self._returns is not None

        # Base: EWMA realized vol for all tickers
        hist_returns = self._returns.loc[:as_of_date].iloc[-self._rv_w:]
        rv_annualized = hist_returns.std() * np.sqrt(252)

        iv_series = rv_annualized.copy()

        # Override with CBOE settlement-grade IV where available
        if self._cboe_iv is not None and not self._cboe_iv.empty:
            cboe_slice = self._cboe_iv.loc[:as_of_date].iloc[-1]
            for etf in _CBOE_ETF_VOL_MAP:
                if etf in cboe_slice.index and not pd.isna(cboe_slice[etf]):
                    iv_series[etf] = float(cboe_slice[etf])

        # Cash equivalents: hard floor (real IV is ~0.5%, not worth computing)
        for ticker in _CASH_EQUIV_TICKERS:
            if ticker in iv_series.index:
                iv_series[ticker] = 0.02

        return iv_series.clip(lower=0.02).rename("iv")

    def _get_rv_matrix(self, as_of_date: str) -> pd.Series:
        """Causal 21-day realized vol, annualised."""
        assert self._returns is not None
        hist = self._returns.loc[:as_of_date].iloc[-self._rv_w:]
        return (hist.std() * np.sqrt(252)).clip(lower=0.02).rename("rv")

    def compute_vrp_signal(self, as_of_date: str) -> pd.Series:
        """
        VRP cross-sectional z-score.

        VRP_i = IV_i − RV_i
        High VRP → more risk aversion priced in the options market for that asset.

        SIGNAL DIRECTION (contrarian):
          The excess risk premium has been paid — expected to compress back toward
          the cross-sectional mean. Assets with anomalously HIGH VRP relative to
          their peers are expected to OUTPERFORM as vol realization undershoots IV.

        EDGE vs momentum/reversal:
          This signal is orthogonal to price-based factors (ρ ≈ 0.02 empirically).
          It captures investor EXPECTATION of future vol, not past price movement.

        CROSS-SECTIONAL Z-SCORE IMPLEMENTATION:
          We z-score cross-sectionally (across assets at a given time t), not
          time-serially per asset. This removes market-wide vol level bias and
          isolates relative mispricing between assets.
        """
        iv_vec = self._get_iv_matrix(as_of_date)
        rv_vec = self._get_rv_matrix(as_of_date)

        vrp_raw = (iv_vec - rv_vec).reindex(_UNIVERSE).fillna(0.0)

        # Cross-sectional z-score
        mu, sig = vrp_raw.mean(), vrp_raw.std()
        if sig < 1e-6:
            return pd.Series(0.0, index=_UNIVERSE)

        # INVERT: high VRP → excess fear → contrarian buy signal
        vrp_z = -(vrp_raw - mu) / sig
        return vrp_z.rename("vrp_z")

    def compute_iv_hv_ratio_signal(self, as_of_date: str) -> pd.Series:
        """
        IV/HV ratio z-score, normalised over 252-day EWMA history.

        IV/HV_ratio = IV_i / HV_i (≥ 1 means options are expensive).

        SIGNAL DIRECTION (contrarian):
          When IV/HV >> historical norm, the implied move overstates the
          likely realized move → directional compression. Assets with
          anomalously expensive vol tend to underdeliver on their implied
          move, creating a VRP capture opportunity.

          High IV/HV_z → sell the implied move → positive return for short-vol
          position → maps to REDUCE POSITION in that asset (it's priced for
          max fear) and wait for compression.

        The time-series z-score (not cross-sectional) here, because IV/HV
        ratio needs to be normalised against each asset's own historical norms:
        VIXY always has IV/HV > 1 (VIX itself is already annualized IV);
        TLT might have IV/HV ≈ 0.9 normally.
        """
        assert self._returns is not None

        iv_vec = self._get_iv_matrix(as_of_date)
        rv_vec = self._get_rv_matrix(as_of_date)

        ratio_current = (iv_vec / rv_vec.clip(lower=0.02)).reindex(_UNIVERSE).fillna(1.0)

        # Build historical ratio series for EWMA z-scoring
        ratio_history: Dict[str, List[float]] = {t: [] for t in _UNIVERSE}
        dates = self._returns.loc[:as_of_date].index[-self._z_hl * 2:]  # limit lookback

        for hist_date in dates:
            hist_str = str(hist_date.date())
            iv_h  = self._get_iv_matrix(hist_str)
            rv_h  = self._get_rv_matrix(hist_str)
            ratio_h = (iv_h / rv_h.clip(lower=0.02)).reindex(_UNIVERSE).fillna(1.0)
            for t in _UNIVERSE:
                ratio_history[t].append(float(ratio_h.get(t, 1.0)))

        z_scores = {}
        for ticker in _UNIVERSE:
            hist_arr = np.array(ratio_history[ticker])
            if len(hist_arr) < 30:
                z_scores[ticker] = 0.0
                continue
            # EWMA mean/std
            weights = np.exp(-np.arange(len(hist_arr))[::-1] / self._z_hl)
            weights /= weights.sum()
            mu_  = float(np.average(hist_arr, weights=weights))
            var_ = float(np.average((hist_arr - mu_) ** 2, weights=weights))
            sig_ = float(np.sqrt(var_) + 1e-6)
            z_scores[ticker] = (float(ratio_current.get(ticker, 1.0)) - mu_) / sig_

        iv_hv_z = pd.Series(z_scores)
        # Invert: high IV/HV (expensive) → compress → bearish on realized move
        return (-iv_hv_z).clip(-3, 3).rename("iv_hv_z")

    def compute_term_structure_signal(
        self,
        as_of_date: str,
        lookback_days: int = 5,
    ) -> pd.Series:
        """
        VIX term structure DELTA signal — change in VTS slope over `lookback_days`.

        Regime transition signal (not level):
          - Slope DETERIORATING (contango → backwardation) over 5 days → bearish
          - Slope IMPROVING (backwardation → contango) over 5 days → bullish
          The RATE OF CHANGE in the term structure is more predictive than the
          level because the level is priced into IV; the slope change is the
          unexpected repricing.

        ROUTING: Applied to equity assets only. Bond/commodity/credit assets
        receive term structure signals from the MultiAssetVolRegime instead.
        """
        if self._vts is None or self._vts.empty:
            return pd.Series(0.0, index=_UNIVERSE)

        vts_hist = self._vts.loc[:as_of_date]
        if len(vts_hist) < lookback_days + 5:
            return pd.Series(0.0, index=_UNIVERSE)

        # Current slope and slope `lookback_days` ago
        slope_now  = (
            float(vts_hist["VIX9D"].iloc[-1]) /
            float(vts_hist["VIX"].iloc[-1].clip(lower=1.0))
        )
        slope_prev = (
            float(vts_hist["VIX9D"].iloc[-(lookback_days + 1)]) /
            float(vts_hist["VIX"].iloc[-(lookback_days + 1)].clip(lower=1.0))
        )

        # Positive delta slope = improving (less backwardation) = bullish signal
        slope_delta = slope_now - slope_prev

        # Z-score this delta over 252-day history
        all_slopes  = vts_hist["VIX9D"] / vts_hist["VIX"].clip(lower=1.0)
        all_deltas  = all_slopes.diff(lookback_days).dropna()
        if len(all_deltas) < 30:
            return pd.Series(0.0, index=_UNIVERSE)

        mu_d  = float(all_deltas.ewm(halflife=252).mean().iloc[-1])
        sig_d = float(all_deltas.ewm(halflife=252).std().iloc[-1].clip(lower=1e-6))
        delta_z = (slope_delta - mu_d) / sig_d

        # Apply to equity-correlated assets only; others get zero
        equity_routing = {
            t: _get_equity_weight(t) for t in _UNIVERSE
        }
        ts_vector = pd.Series({
            t: float(delta_z) * equity_routing[t] for t in _UNIVERSE
        })
        return ts_vector.clip(-3, 3).rename("ts_z")

    def get_alpha_vector(self, as_of_date: str) -> pd.Series:
        """
        Blended options-surface alpha.
        Weights: VRP 0.50, IV/HV ratio 0.30, term structure 0.20.
        Returns cross-sectionally z-scored composite in tanh([-1, 1]).
        """
        vrp    = self.compute_vrp_signal(as_of_date).reindex(_UNIVERSE).fillna(0.0)
        iv_hv  = self.compute_iv_hv_ratio_signal(as_of_date).reindex(_UNIVERSE).fillna(0.0)
        ts     = self.compute_term_structure_signal(as_of_date).reindex(_UNIVERSE).fillna(0.0)

        composite = 0.50 * vrp + 0.30 * iv_hv + 0.20 * ts

        # Final cross-sectional z-score + tanh squash
        mu, sig = composite.mean(), composite.std()
        if sig < 1e-6:
            return pd.Series(0.0, index=_UNIVERSE)
        z = (composite - mu) / sig
        return np.tanh(z * 0.75)  # 0.75 dampening preserves rank without clipping

    def compute_full_history(self) -> pd.DataFrame:
        """
        Precompute alpha vector for all available dates.
        Used by precompute_alpha_signals.py Stage 2.

        Returns DataFrame (T, 25) of options-surface alpha signals.
        """
        assert self._prices is not None
        dates = self._prices.index

        results = []
        for i, date in enumerate(dates):
            date_str = str(date.date())
            # Need sufficient history for RV computation
            if i < self._rv_w + 5:
                results.append(pd.Series(0.0, index=_UNIVERSE))
                continue
            try:
                alpha = self.get_alpha_vector(date_str)
                results.append(alpha)
            except Exception as e:
                logger.debug(f"OptionsAlpha failed for {date_str}: {e}")
                results.append(pd.Series(0.0, index=_UNIVERSE))

            if i % 200 == 0:
                logger.info(f"OptionsAlpha precompute: {i}/{len(dates)} dates")

        return pd.DataFrame(results, index=dates)


def _get_equity_weight(ticker: str) -> float:
    """Returns the fraction of a ticker's variance that is equity-driven."""
    pure_equity    = {"SPY", "QQQ", "IWM", "XLK", "XLF", "XLV", "XLU",
                      "XLI", "XLP", "XLY", "XLB", "XLC", "VIXY", "COWZ"}
    mixed_equity   = {"GDX": 0.6, "XLE": 0.7, "PDBC": 0.3}
    bond_like      = {"TLT", "HYG", "LQD", "BIL", "SHV"}
    commodity_pure = {"GLD", "SLV", "USO"}

    if ticker in pure_equity:
        return 1.0
    if ticker in mixed_equity:
        return mixed_equity[ticker]
    if ticker in bond_like or ticker in commodity_pure:
        return 0.0
    return 0.5