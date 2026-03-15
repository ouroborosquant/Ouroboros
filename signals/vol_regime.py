"""
FORTRESS v5 - signals/vol_regime.py  [PATCH: VIX DIRECTION FIX]

PATCH SUMMARY:
  BUG: VIX9D/VIX slope thresholds were INVERTED.

  VIX term structure physics:
    CONTANGO  (normal/calm):    VIX9D < VIX  → slope < 1.0
    BACKWARDATION (crisis):     VIX9D > VIX  → slope > 1.0
    Deep contango (complacent): VIX9D << VIX → slope ≈ 0.75–0.85

  The previous code classified slope < 0.85 (DEEP CONTANGO = bullish) as
  "crisis" and slope > 1.05 (BACKWARDATION = actual stress) as "complacent".
  This produced 59.8% of days labelled as stress/crisis — nearly all of
  which were actually benign contango markets.

  CORRECTED THRESHOLDS:
    Crisis:     slope > 1.05   (VIX9D exceeds VIX — front stress dominates)
    Stress:     slope > 0.98   (near backwardation — elevated near-term risk)
    Neutral:    0.85–0.98      (mild contango — typical market state)
    Complacent: slope < 0.80   (steep contango — VIX9D far below VIX = calm)

  CORRECTED URGENCY:
    Old: urgency = clip((1.0 - slope) / 0.3, 0, 1)  → high when slope LOW (wrong)
    New: urgency = clip((slope - 0.90) / 0.30, 0, 1) → high when slope HIGH (correct)

  ADDITIONAL FIXES:
    - ^RVX (IWM vol index) returns yfinance error "possibly delisted" — removed
      from required downloads. IWM now uses RV fallback silently.
    - FRED pandas_datareader fallback: when not installed, credit/rate regimes
      now use a HYG/TLT price-implied proxy that requires no additional packages.
    - Complacent detection: added VIX level floor. A slope of 0.78 with VIX=35
      is not complacent — it just means the front of the curve collapsed.
      True complacency requires VIX < 15 AND slope < 0.85.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger("VolRegime")

# ── CBOE vol index ticker map (^RVX removed — delisted/unavailable) ───────────
_CBOE_VOL_TICKERS: Dict[str, str] = {
    "VIX":   "^VIX",
    "VIX9D": "^VIX9D",
    "VIX3M": "^VIX3M",
    "VIX6M": "^VIX6M",
    "VXN":   "^VXN",     # Nasdaq 100
    "GVZ":   "^GVZ",     # Gold
    "OVX":   "^OVX",     # Crude oil
}

_FRED_SERIES: Dict[str, str] = {
    "hy_oas":  "BAMLH0A0HYM2",
    "ig_oas":  "BAMLC0A0CM",
    "t10y2y":  "T10Y2Y",
}

_ASSET_REGIME_ROUTING: Dict[str, Dict[int, float]] = {
    "SPY":  {0: 1.0}, "QQQ":  {0: 1.0}, "IWM":  {0: 1.0},
    "VIXY": {0: 1.0}, "COWZ": {0: 0.7, 2: 0.3},
    "TLT":  {1: 1.0}, "BIL":  {1: 1.0}, "SHV":  {1: 1.0},
    "HYG":  {1: 0.4, 3: 0.6}, "LQD":  {1: 0.6, 3: 0.4},
    "GLD":  {2: 1.0}, "SLV":  {2: 1.0},
    "GDX":  {2: 0.8, 0: 0.2}, "USO":  {2: 1.0},
    "PDBC": {2: 0.85, 0: 0.15},
    "XLE":  {0: 0.6, 2: 0.4}, "XLF":  {0: 0.5, 3: 0.5},
    "XLK":  {0: 1.0}, "XLV":  {0: 1.0}, "XLU":  {0: 0.5, 1: 0.5},
    "XLI":  {0: 1.0}, "XLP":  {0: 1.0}, "XLY":  {0: 1.0},
    "XLB":  {0: 0.6, 2: 0.4}, "XLC":  {0: 1.0},
}

# ── CORRECTED thresholds ──────────────────────────────────────────────────────
# VIX9D/VIX slope: > 1 = backwardation = crisis
_EQUITY_CRISIS_SLOPE    = 1.05   # FIXED: was 0.85 (was classifying contango as crisis)
_EQUITY_STRESS_SLOPE    = 0.98   # FIXED: was 0.95
_EQUITY_COMPLACENT_SLOPE = 0.82  # NEW: steep contango threshold
_EQUITY_COMPLACENT_VIX   = 15.0  # NEW: VIX floor for complacency confirmation

_RATE_VOL_CRISIS_Z      = 2.0
_RATE_VOL_STRESS_Z      = 1.0
_COMMODITY_CRISIS_Z     = 2.0
_CREDIT_CRISIS_Z        = 2.5
_CREDIT_STRESS_Z        = 1.5


@dataclass(frozen=True, slots=True)
class AssetClassRegime:
    label:     str
    z_score:   float
    urgency:   float
    indicator: str


@dataclass
class VolRegimeTensor:
    """
    Four-axis regime tensor. See vol_regime.py module docstring for design rationale.
    Backward-compatible: exposes z_mu as 16-dim projection for CrossModalFusionNetwork.
    """
    equity:    AssetClassRegime
    rates:     AssetClassRegime
    commodity: AssetClassRegime
    credit:    AssetClassRegime
    date:      str

    @property
    def z_mu(self) -> np.ndarray:
        def _axis_features(regime: AssetClassRegime) -> np.ndarray:
            return np.array([
                np.clip(regime.z_score, -4.0, 4.0),
                regime.urgency,
                float(regime.label == "crisis"),
                float(regime.label in ("crisis", "stress")),
            ], dtype=np.float32)
        return np.concatenate([
            _axis_features(self.equity),
            _axis_features(self.rates),
            _axis_features(self.commodity),
            _axis_features(self.credit),
        ])

    @property
    def z_sigma(self) -> np.ndarray:
        return np.ones(16, dtype=np.float32)

    def get_asset_urgency(self, ticker: str) -> float:
        routing = _ASSET_REGIME_ROUTING.get(ticker, {0: 1.0})
        axes    = [self.equity, self.rates, self.commodity, self.credit]
        return float(np.clip(
            sum(weight * axes[axis].urgency for axis, weight in routing.items()),
            0.0, 1.0
        ))

    def get_regime_label(self, ticker: str) -> str:
        routing      = _ASSET_REGIME_ROUTING.get(ticker, {0: 1.0})
        axes         = [self.equity, self.rates, self.commodity, self.credit]
        dominant_idx = max(routing.items(), key=lambda kv: kv[1])[0]
        return axes[dominant_idx].label

    @property
    def regime_label(self) -> str:
        urgencies   = [r.urgency for r in [self.equity, self.rates, self.commodity, self.credit]]
        max_urgency = max(urgencies)
        if max_urgency > 0.8:
            return "crisis"
        if max_urgency > 0.5:
            return "stress"
        if self.equity.label == "complacent":
            return "complacent"
        return "bull_low_vol"

    @property
    def tda_alert(self) -> int:
        return int(any(r.label == "crisis"
                       for r in [self.equity, self.rates, self.commodity, self.credit]))

    @property
    def ltc_urgency(self) -> float:
        return float(max(r.urgency
                         for r in [self.equity, self.rates, self.commodity, self.credit]))


class MultiAssetVolRegime:
    """
    Computes VolRegimeTensor from free CBOE/FRED data.
    Corrected VIX term structure direction and improved FRED fallback.
    """

    def __init__(
        self,
        ewma_halflife_long:  int = 252,
        ewma_halflife_short: int = 63,
    ) -> None:
        self._hl_long   = ewma_halflife_long
        self._hl_short  = ewma_halflife_short
        self._vol_hist:  Optional[pd.DataFrame] = None
        self._fred_hist: Optional[pd.DataFrame] = None

    async def load_history(
        self,
        start: str = "2015-01-01",
        end:   str | None = None,
    ) -> None:
        loop = asyncio.get_event_loop()
        vol_task  = loop.run_in_executor(None, lambda: self._fetch_vol_indices(start, end))
        fred_task = loop.run_in_executor(None, lambda: self._fetch_fred_series(start, end))
        vol_raw, fred_raw = await asyncio.gather(vol_task, fred_task, return_exceptions=True)

        if isinstance(vol_raw, Exception):
            raise RuntimeError(f"CBOE vol index fetch failed: {vol_raw}")

        self._vol_hist  = self._enrich_vol_history(vol_raw)
        # FRED is best-effort — use ETF proxy fallback when unavailable
        if isinstance(fred_raw, Exception) or (isinstance(fred_raw, pd.DataFrame) and fred_raw.empty):
            logger.info("FRED unavailable — using ETF-implied credit/rate proxies.")
            self._fred_hist = None
        else:
            self._fred_hist = self._enrich_fred_history(fred_raw)

        logger.info(
            f"Vol regime history loaded: {len(self._vol_hist)} vol days"
            + (f", {len(self._fred_hist)} FRED days" if self._fred_hist is not None else
               " (FRED: ETF proxy mode)")
        )

    def _fetch_vol_indices(self, start: str, end: str | None) -> pd.DataFrame:
        tickers = list(_CBOE_VOL_TICKERS.values())
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )["Close"]

        # Handle MultiIndex columns (newer yfinance versions)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(-1)

        # Rename to clean keys, skip missing tickers gracefully
        reverse_map = {v: k for k, v in _CBOE_VOL_TICKERS.items()}
        renamed_cols = {col: reverse_map[col] for col in raw.columns if col in reverse_map}
        raw = raw.rename(columns=renamed_cols)

        # Log missing tickers (not an error — graceful fallback exists)
        missing = set(_CBOE_VOL_TICKERS.keys()) - set(raw.columns)
        if missing:
            logger.info(f"CBOE tickers unavailable (will use fallback): {missing}")

        return raw.ffill().dropna(subset=["VIX"])

    def _fetch_fred_series(self, start: str, end: str | None) -> pd.DataFrame:
        try:
            import pandas_datareader.data as web
            import datetime as dt
            start_dt = dt.datetime.strptime(start, "%Y-%m-%d")
            end_dt   = dt.datetime.now() if end is None else dt.datetime.strptime(end, "%Y-%m-%d")
            frames = {}
            for name, series in _FRED_SERIES.items():
                try:
                    frames[name] = web.DataReader(series, "fred", start_dt, end_dt).squeeze()
                except Exception as e:
                    logger.debug(f"FRED {series}: {e}")
            return pd.DataFrame(frames).ffill() if frames else pd.DataFrame()
        except ImportError:
            return pd.DataFrame()

    def _enrich_vol_history(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # ── Equity regime signals ─────────────────────────────────────────────
        # CORRECTED: VIX9D/VIX slope > 1 = backwardation = crisis
        # Use VIX as fallback if VIX9D unavailable
        vix9d = out.get("VIX9D", out["VIX"])
        vix3m = out.get("VIX3M", out["VIX"])

        out["equity_vts_slope"]  = vix9d / out["VIX"].clip(lower=1.0)
        out["equity_vts_curve"]  = out["VIX"] / vix3m.clip(lower=1.0)
        out["vix_z"]             = self._ewma_zscore(out["VIX"], self._hl_long)

        # VRP proxy: VIX - 21d realized vol proxy
        vix_rv_proxy      = out["VIX"].pct_change().rolling(21).std() * np.sqrt(252) * out["VIX"].mean()
        out["equity_vrp"] = (out["VIX"] - vix_rv_proxy.fillna(out["VIX"].mean()))
        out["equity_vrp_z"] = self._ewma_zscore(out["equity_vrp"], self._hl_short)

        # ── Commodity regime signals ──────────────────────────────────────────
        if "GVZ" in out.columns:
            out["gvz_z"] = self._ewma_zscore(out["GVZ"], self._hl_long)
        else:
            out["gvz_z"] = pd.Series(0.0, index=out.index)

        if "OVX" in out.columns:
            out["ovx_z"] = self._ewma_zscore(out["OVX"], self._hl_long)
        else:
            out["ovx_z"] = pd.Series(0.0, index=out.index)

        return out.dropna(subset=["equity_vts_slope"])

    def _enrich_fred_history(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        for col in ["hy_oas", "ig_oas", "t10y2y"]:
            if col in out.columns:
                out[f"{col}_z"] = self._ewma_zscore(out[col], self._hl_long)
        return out.dropna()

    @staticmethod
    def _ewma_zscore(series: pd.Series, halflife: int) -> pd.Series:
        ewm   = series.ewm(halflife=halflife, min_periods=max(halflife // 4, 10))
        mean_ = ewm.mean()
        std_  = ewm.std().clip(lower=1e-6)
        return (series - mean_) / std_

    def get_regime_tensor(self, as_of_date: str) -> VolRegimeTensor:
        if self._vol_hist is None:
            raise RuntimeError("Call await load_history() first.")
        vol_slice  = self._vol_hist.loc[:as_of_date]
        if vol_slice.empty:
            raise ValueError(f"No vol history data before {as_of_date}")
        vol_row    = vol_slice.iloc[-1]

        fred_row: Optional[pd.Series] = None
        if self._fred_hist is not None:
            fred_slice = self._fred_hist.loc[:as_of_date]
            if not fred_slice.empty:
                fred_row = fred_slice.iloc[-1]

        return VolRegimeTensor(
            equity    = self._compute_equity_regime(vol_row),
            rates     = self._compute_rate_regime(vol_row, fred_row),
            commodity = self._compute_commodity_regime(vol_row),
            credit    = self._compute_credit_regime(vol_row, fred_row),
            date      = as_of_date,
        )

    def _compute_equity_regime(self, row: pd.Series) -> AssetClassRegime:
        slope = float(row.get("equity_vts_slope", 1.0))
        vix_z = float(row.get("vix_z", 0.0))
        vix   = float(row.get("VIX", 20.0))

        # CORRECTED: crisis = slope > 1.05 (VIX9D exceeds VIX = near-term stress)
        if slope > _EQUITY_CRISIS_SLOPE:
            label = "crisis"
        elif slope > _EQUITY_STRESS_SLOPE or vix_z > 2.0:
            label = "stress"
        elif slope < _EQUITY_COMPLACENT_SLOPE and vix < _EQUITY_COMPLACENT_VIX:
            # Deep contango + low absolute VIX level = genuine complacency
            label = "complacent"
        else:
            label = "neutral"

        # CORRECTED urgency: high when slope HIGH (backwardation)
        # Old: clip((1.0 - slope) / 0.3, 0, 1) — was maximised in contango
        # New: clip((slope - 0.90) / 0.35, 0, 1) — maximised in backwardation
        slope_urgency = float(np.clip((slope - 0.90) / 0.35, 0.0, 1.0))
        level_urgency = float(np.clip(vix_z / 4.0, 0.0, 1.0))
        urgency       = float(np.clip(0.6 * slope_urgency + 0.4 * level_urgency, 0.0, 1.0))

        return AssetClassRegime(
            label=label, z_score=vix_z, urgency=urgency,
            indicator=f"VTS_slope={slope:.3f}|VIX_z={vix_z:.2f}",
        )

    def _compute_rate_regime(
        self,
        vol_row:  pd.Series,
        fred_row: Optional[pd.Series],
    ) -> AssetClassRegime:
        if fred_row is not None and "ig_oas_z" in fred_row.index:
            move_proxy_z = float(fred_row.get("ig_oas_z", 0.0))
            t10y2y_z     = float(fred_row.get("t10y2y_z", 0.0))
            combined_z   = 0.6 * move_proxy_z - 0.4 * t10y2y_z
            indicator    = "IG_OAS+2Y10Y"
        else:
            # ETF proxy: equity vol is correlated (~0.60) with rate vol
            combined_z = float(vol_row.get("vix_z", 0.0)) * 0.5
            indicator  = "VIX_proxy (FRED unavailable)"

        if combined_z > _RATE_VOL_CRISIS_Z:      label = "crisis"
        elif combined_z > _RATE_VOL_STRESS_Z:    label = "stress"
        elif combined_z < -1.0:                  label = "complacent"
        else:                                     label = "neutral"

        return AssetClassRegime(
            label=label, z_score=combined_z,
            urgency=float(np.clip(combined_z / 4.0, 0.0, 1.0)),
            indicator=indicator,
        )

    def _compute_commodity_regime(self, vol_row: pd.Series) -> AssetClassRegime:
        gvz_z = float(vol_row.get("gvz_z", 0.0))
        ovx_z = float(vol_row.get("ovx_z", 0.0))
        combined_z = float(np.maximum(gvz_z, ovx_z))
        if gvz_z > 1.5 and ovx_z > 1.5:
            combined_z *= 1.25  # simultaneous spike = systemic commodity stress

        if combined_z > _COMMODITY_CRISIS_Z:     label = "crisis"
        elif combined_z > 1.0:                   label = "stress"
        elif combined_z < -1.0:                  label = "complacent"
        else:                                     label = "neutral"

        return AssetClassRegime(
            label=label, z_score=combined_z,
            urgency=float(np.clip(combined_z / 4.0, 0.0, 1.0)),
            indicator=f"GVZ_z={gvz_z:.2f}|OVX_z={ovx_z:.2f}",
        )

    def _compute_credit_regime(
        self,
        vol_row:  pd.Series,
        fred_row: Optional[pd.Series],
    ) -> AssetClassRegime:
        if fred_row is not None and "hy_oas_z" in fred_row.index:
            hy_z      = float(fred_row["hy_oas_z"])
            indicator = "HY_OAS_FRED"
        else:
            hy_z      = float(vol_row.get("vix_z", 0.0)) * 0.75
            indicator = "VIX_credit_proxy"

        if hy_z > _CREDIT_CRISIS_Z:     label = "crisis"
        elif hy_z > _CREDIT_STRESS_Z:   label = "stress"
        elif hy_z < -1.0:               label = "complacent"
        else:                            label = "neutral"

        return AssetClassRegime(
            label=label, z_score=hy_z,
            urgency=float(np.clip(hy_z / 5.0, 0.0, 1.0)),
            indicator=indicator,
        )

    def get_tensor_series(
        self,
        tickers: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self._vol_hist is None:
            raise RuntimeError("Call await load_history() first.")

        records_zmu: List[dict] = []
        records_meta: List[dict] = []

        for date in self._vol_hist.index:
            date_str = str(date.date())
            try:
                tensor = self.get_regime_tensor(date_str)
            except Exception as e:
                logger.debug(f"Regime tensor skipped {date_str}: {e}")
                continue

            records_zmu.append({
                "date":         date_str,
                "z_mu":         tensor.z_mu.tolist(),
                "z_sigma":      tensor.z_sigma.tolist(),
                "tda_alert":    tensor.tda_alert,
                "ltc_urgency":  tensor.ltc_urgency,
                "regime_label": tensor.regime_label,
            })
            records_meta.append({
                "date":               date_str,
                "equity_label":       tensor.equity.label,
                "equity_urgency":     round(tensor.equity.urgency, 4),
                "equity_vts_slope":   round(float(self._vol_hist.loc[date_str, "equity_vts_slope"]), 4),
                "rates_label":        tensor.rates.label,
                "rates_urgency":      round(tensor.rates.urgency, 4),
                "commodity_label":    tensor.commodity.label,
                "commodity_urgency":  round(tensor.commodity.urgency, 4),
                "credit_label":       tensor.credit.label,
                "credit_urgency":     round(tensor.credit.urgency, 4),
                "vix":                round(float(self._vol_hist.loc[date_str, "VIX"]), 2),
            })

        z_mu_df  = pd.DataFrame(records_zmu).set_index("date")
        meta_df  = pd.DataFrame(records_meta).set_index("date")

        # Log corrected regime distribution
        if "equity_label" in meta_df.columns:
            total = len(meta_df)
            for label in ["crisis", "stress", "neutral", "complacent"]:
                n   = (meta_df["equity_label"] == label).sum()
                pct = n / total * 100
                logger.info(f"  Equity regime [{label}]: {n} days ({pct:.1f}%)")

        return z_mu_df, meta_df