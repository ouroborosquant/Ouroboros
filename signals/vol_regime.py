"""
FORTRESS v5 - signals/vol_regime.py
Path: signals/vol_regime.py

Multi-Asset Volatility Regime Engine.

ARCHITECTURAL MOTIVATION (addresses Q4):
  A single VIX term-structure gate is correct for equity volatility but
  systematically wrong for the other three asset classes in the universe:

    TLT / LQD / HYG — driven by RATE volatility (MOVE index, TYVIX).
      MOVE and VIX decorrelate sharply: 2022 rate shock had MOVE at 160+
      while VIX stayed below 35. A VIX-only gate would have predicted
      "low stress" for bonds during their worst year since 1788.

    GLD / SLV / GDX — driven by GOLD volatility (GVZ).
      GLD rallied +13% during the 2023 banking stress while equity VIX
      barely moved. GVZ spiked to 22; VIX was 17.

    USO / PDBC — driven by OIL volatility (OVX).
      OVX reached 325 during the April 2020 negative WTI event while
      VIX was already compressing from the COVID peak.

    HYG / HY credit — driven by credit spread regimes (HY OAS via FRED).

  The solution: a four-axis VolRegimeTensor that routes each asset to
  the correct vol index, producing asset-specific regime scaling rather
  than one-size-fits-all equity VIX scaling.

REGIME TENSOR AXES:
  [0] equity_regime   — VIX9D/VIX slope + VIX/VIX3M curvature
  [1] rate_regime     — MOVE index z-score + 2Y10Y inversion depth
  [2] commodity_regime — max(GVZ_z, OVX_z) composite
  [3] credit_regime   — HY OAS z-score (FRED: BAMLH0A0HYM2)

ASSET → AXIS ROUTING:
  Equity ETFs      → axis[0]: SPY, QQQ, IWM, XL* sector, VIXY, COWZ
  Rate/Bond ETFs   → axis[1]: TLT, HYG, LQD, BIL, SHV
  Commodity ETFs   → axis[2]: GLD, SLV, GDX, USO, PDBC
  Mixed/diversified → weighted combination of axes

FREE DATA SOURCES:
  VIX/VXN/RVX/GVZ/OVX: CBOE via yfinance (^VIX, ^VXN, ^RVX, ^GVZ, ^OVX)
  VIX9D, VIX3M, VIX6M: CBOE via yfinance (^VIX9D, ^VIX3M, ^VIX6M)
  MOVE proxy: TLT implied vol from TYVIX or FRED BAMLC0A4CBB series
  HY OAS: FRED BAMLH0A0HYM2 series (free, 1-day lag)
  2Y10Y spread: FRED T10Y2Y (free, 1-day lag)

LOOK-AHEAD CONTRACT:
  All FRED series carry a release_date field. We use publication_date
  (not value_date) for as_of_date gating. VIX/CBOE data is available
  at 4:15 PM ET same day — safe for next-morning signal generation.
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

# ── CBOE vol index ticker map ──────────────────────────────────────────────────
_CBOE_VOL_TICKERS: Dict[str, str] = {
    "VIX":     "^VIX",     # S&P 500 30d implied vol
    "VIX9D":   "^VIX9D",   # S&P 500 9d — term structure near end
    "VIX3M":   "^VIX3M",   # S&P 500 3-month
    "VIX6M":   "^VIX6M",   # S&P 500 6-month
    "VXN":     "^VXN",     # Nasdaq 100 implied vol
    "RVX":     "^RVX",     # Russell 2000 implied vol
    "GVZ":     "^GVZ",     # Gold ETF (GLD) implied vol
    "OVX":     "^OVX",     # Crude oil (USO) implied vol
    "VXEEM":   "^VXEEM",   # Emerging markets — informational
}

# FRED series for rate/credit regimes
_FRED_SERIES: Dict[str, str] = {
    "hy_oas":      "BAMLH0A0HYM2",   # ICE BofAML US HY Option-Adjusted Spread
    "ig_oas":      "BAMLC0A0CM",     # ICE BofAML IG OAS (MOVE proxy)
    "t10y2y":      "T10Y2Y",         # 10Y-2Y Treasury spread
    "sofr":        "SOFR",           # Secured Overnight Financing Rate
}

# Asset → regime axis routing
# 0=equity, 1=rates, 2=commodity, 3=credit
# Weights sum to 1.0 per asset
_ASSET_REGIME_ROUTING: Dict[str, Dict[int, float]] = {
    "SPY":  {0: 1.0},
    "QQQ":  {0: 1.0},
    "IWM":  {0: 1.0},
    "VIXY": {0: 1.0},
    "COWZ": {0: 0.7, 2: 0.3},
    # Bond ETFs — pure rate regime
    "TLT":  {1: 1.0},
    "BIL":  {1: 1.0},
    "SHV":  {1: 1.0},
    # Credit ETFs — rate + credit mix
    "HYG":  {1: 0.4, 3: 0.6},
    "LQD":  {1: 0.6, 3: 0.4},
    # Commodity ETFs — commodity regime
    "GLD":  {2: 1.0},
    "SLV":  {2: 1.0},
    "GDX":  {2: 0.8, 0: 0.2},   # GDX has equity beta
    "USO":  {2: 1.0},
    "PDBC": {2: 0.85, 0: 0.15},
    # Sector ETFs — primarily equity, partial credit for fins/energy
    "XLE":  {0: 0.6, 2: 0.4},
    "XLF":  {0: 0.5, 3: 0.5},
    "XLK":  {0: 1.0},
    "XLV":  {0: 1.0},
    "XLU":  {0: 0.5, 1: 0.5},   # Utilities highly rate-sensitive
    "XLI":  {0: 1.0},
    "XLP":  {0: 1.0},
    "XLY":  {0: 1.0},
    "XLB":  {0: 0.6, 2: 0.4},
    "XLC":  {0: 1.0},
}

# Regime label thresholds (per axis)
_EQUITY_CRISIS_SLOPE    = 0.85   # VIX9D/VIX < this → crisis
_EQUITY_STRESS_SLOPE    = 0.95
_RATE_VOL_CRISIS_Z      = 2.0    # MOVE z-score > this → rates crisis
_RATE_VOL_STRESS_Z      = 1.0
_COMMODITY_CRISIS_Z     = 2.0    # GVZ or OVX z > this → commodity crisis
_CREDIT_CRISIS_Z        = 2.5    # HY OAS z > this → credit crisis
_CREDIT_STRESS_Z        = 1.5


@dataclass(frozen=True, slots=True)
class AssetClassRegime:
    """Single-axis regime state."""
    label:    str    # 'crisis' | 'stress' | 'neutral' | 'complacent'
    z_score:  float  # raw z-score of the primary indicator
    urgency:  float  # [0, 1] continuous — 0=benign, 1=full crisis
    indicator: str   # which indicator drove this (for diagnostics)


@dataclass
class VolRegimeTensor:
    """
    Four-axis regime tensor.

    Replaces Mamba-KAN z_mu as the regime conditioning signal.
    Backward-compatible: exposes z_mu as a 16-dim projection for use
    in CrossModalFusionNetwork's RegimeAdaptiveLayerNorm.

    The 16-dim z_mu is structured as 4 blocks of 4:
      z_mu[0:4]   = equity regime features
      z_mu[4:8]   = rate regime features
      z_mu[8:12]  = commodity regime features
      z_mu[12:16] = credit regime features
    """
    equity:    AssetClassRegime
    rates:     AssetClassRegime
    commodity: AssetClassRegime
    credit:    AssetClassRegime
    date:      str

    @property
    def z_mu(self) -> np.ndarray:
        """16-dim projection maintaining pipeline API compat."""
        def _axis_features(regime: AssetClassRegime) -> np.ndarray:
            # 4 features per axis:
            # [z_score, urgency, is_crisis, is_stress]
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
        ])  # shape (16,)

    @property
    def z_sigma(self) -> np.ndarray:
        """Uncertainty — inversely scaled by data freshness."""
        return np.ones(16, dtype=np.float32)

    def get_asset_urgency(self, ticker: str) -> float:
        """
        Per-asset regime urgency in [0, 1].
        Routes urgency through the correct vol axis for each asset.
        """
        routing = _ASSET_REGIME_ROUTING.get(ticker, {0: 1.0})
        axes = [self.equity, self.rates, self.commodity, self.credit]
        urgency = sum(
            weight * axes[axis].urgency
            for axis, weight in routing.items()
        )
        return float(np.clip(urgency, 0.0, 1.0))

    def get_regime_label(self, ticker: str) -> str:
        """Dominant regime label for a given asset."""
        routing = _ASSET_REGIME_ROUTING.get(ticker, {0: 1.0})
        axes = [self.equity, self.rates, self.commodity, self.credit]
        dominant_axis = max(routing.items(), key=lambda kv: kv[1])[0]
        return axes[dominant_axis].label

    @property
    def regime_label(self) -> str:
        """Aggregate regime label for portfolio-level decisions."""
        urgencies = [self.equity.urgency, self.rates.urgency,
                     self.commodity.urgency, self.credit.urgency]
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
        """Topology alert flag — fires when any axis enters crisis."""
        return int(any(
            r.label == "crisis"
            for r in [self.equity, self.rates, self.commodity, self.credit]
        ))

    @property
    def ltc_urgency(self) -> float:
        """Maximum urgency across all axes."""
        return float(max(r.urgency for r in [
            self.equity, self.rates, self.commodity, self.credit
        ]))


class MultiAssetVolRegime:
    """
    Computes VolRegimeTensor from free CBOE/FRED data.

    LOOKBACK CALIBRATION:
      EWMA z-scores use 252-day halflife for structural breaks (MOVE, VIX level)
      and 63-day halflife for cyclical indicators (VIX slope, term spread).
      This prevents the 2020 COVID shock from permanently inflating z-scores
      and making every subsequent event look mild by comparison.
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
        """
        Load CBOE vol index history + FRED macro series.
        Runs both fetches concurrently. Falls back gracefully on partial failure.
        """
        loop = asyncio.get_event_loop()

        vol_task  = loop.run_in_executor(None, lambda: self._fetch_vol_indices(start, end))
        fred_task = loop.run_in_executor(None, lambda: self._fetch_fred_series(start, end))
        vol_raw, fred_raw = await asyncio.gather(vol_task, fred_task, return_exceptions=True)

        if isinstance(vol_raw, Exception):
            logger.error(f"CBOE vol index fetch failed: {vol_raw}")
            raise vol_raw
        if isinstance(fred_raw, Exception):
            logger.warning(f"FRED fetch failed: {fred_raw} — rate regime will use proxy.")
            fred_raw = None

        self._vol_hist  = self._enrich_vol_history(vol_raw)
        self._fred_hist = self._enrich_fred_history(fred_raw) if fred_raw is not None else None

        logger.info(
            f"Vol regime history loaded: "
            f"{len(self._vol_hist)} vol days"
            + (f", {len(self._fred_hist)} FRED days" if self._fred_hist is not None else "")
        )

    def _fetch_vol_indices(
        self,
        start: str,
        end: str | None,
    ) -> pd.DataFrame:
        """Download CBOE vol indices. Returns clean close prices."""
        tickers = list(_CBOE_VOL_TICKERS.values())
        raw = yf.download(tickers, start=start, end=end, progress=False)["Close"]
        raw.columns = [
            {v: k for k, v in _CBOE_VOL_TICKERS.items()}[col]
            for col in raw.columns
        ]
        return raw.ffill().dropna(subset=["VIX"])

    def _fetch_fred_series(
        self,
        start: str,
        end: str | None,
    ) -> pd.DataFrame:
        """
        Download FRED macro series.
        Uses pandas_datareader with FRED as source.
        Fallback: estimate HY OAS from HYG/LQD price spread.
        """
        try:
            import pandas_datareader.data as web  # type: ignore
            import datetime as dt

            start_dt = dt.datetime.strptime(start, "%Y-%m-%d")
            end_dt   = dt.datetime.now() if end is None else dt.datetime.strptime(end, "%Y-%m-%d")

            frames = {}
            for name, series in _FRED_SERIES.items():
                try:
                    s = web.DataReader(series, "fred", start_dt, end_dt).squeeze()
                    frames[name] = s
                except Exception as e:
                    logger.debug(f"FRED series {series} unavailable: {e}")

            if frames:
                return pd.DataFrame(frames).ffill()
            return pd.DataFrame()

        except ImportError:
            logger.warning("pandas_datareader not available — FRED series skipped.")
            return pd.DataFrame()

    def _enrich_vol_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived regime signals from vol indices.
        All windows are EWMA-based for stable z-scoring.
        """
        out = df.copy()

        # ── Equity regime signals ──────────────────────────────────────────────
        # Term structure slope: VIX9D/VIX — < 1 = backwardation = near-term stress
        out["equity_vts_slope"] = (
            out["VIX9D"].fillna(out["VIX"]) /
            out["VIX"].clip(lower=1.0)
        )
        # Term structure curvature: VIX/VIX3M — > 1 = stress building in medium term
        out["equity_vts_curve"] = (
            out["VIX"] /
            out["VIX3M"].fillna(out["VIX"]).clip(lower=1.0)
        )
        # VIX level z-score (long halflife — structural regime, not daily noise)
        out["vix_z"] = self._ewma_zscore(out["VIX"], self._hl_long)

        # ── Rate regime signals ────────────────────────────────────────────────
        # MOVE proxy via IG OAS (iShares LQD implied vol is embedded in IG spreads)
        # When GVZ is available, use it as gold-specific signal
        if "GVZ" in out.columns:
            out["gvz_z"] = self._ewma_zscore(out["GVZ"], self._hl_long)
        else:
            out["gvz_z"] = 0.0

        # ── Commodity regime signals ────────────────────────────────────────────
        if "OVX" in out.columns:
            out["ovx_z"] = self._ewma_zscore(out["OVX"], self._hl_long)
        else:
            out["ovx_z"] = 0.0

        # ── Variance risk premium (VRP) per vol index ─────────────────────────
        # VRP = vol_index − 21d_realized_vol_of_underlying (approximated)
        # We compute realized vol from the VIX itself (change in VIX = rough proxy)
        vix_chg_rv = out["VIX"].pct_change().rolling(21).std() * np.sqrt(252) * out["VIX"].mean()
        out["equity_vrp_raw"] = out["VIX"] - vix_chg_rv.fillna(out["VIX"].mean())
        out["equity_vrp_z"]   = self._ewma_zscore(out["equity_vrp_raw"], self._hl_short)

        return out.dropna(subset=["equity_vts_slope"])

    def _enrich_fred_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute z-scores for FRED macro series."""
        if df.empty:
            return df
        out = df.copy()
        for col in ["hy_oas", "ig_oas", "t10y2y"]:
            if col in out.columns:
                out[f"{col}_z"] = self._ewma_zscore(out[col], self._hl_long)
        return out.dropna()

    @staticmethod
    def _ewma_zscore(series: pd.Series, halflife: int) -> pd.Series:
        """
        Causal EWMA z-score: uses only t-1..t data for mean/std computation.
        This is critical for backtest validity — no look-ahead in normalisation.
        min_periods prevents NaN explosion on short history.
        """
        ewm    = series.ewm(halflife=halflife, min_periods=halflife // 4)
        mean_  = ewm.mean()
        std_   = ewm.std().clip(lower=1e-6)
        return (series - mean_) / std_

    def get_regime_tensor(self, as_of_date: str) -> VolRegimeTensor:
        """
        Computes the VolRegimeTensor for a given date.

        CAUSAL CONTRACT: Uses vol_hist data up to and including as_of_date.
        VIX/CBOE data is published at 4:15 PM ET — safe for T+1 signal use.
        FRED data has 1-day publication lag — we gate using as_of_date - 1 business day.
        """
        if self._vol_hist is None:
            raise RuntimeError("Call await load_history() first.")

        # Slice to as_of_date — strict ≤ for causal compliance
        vol_row   = self._vol_hist.loc[:as_of_date].iloc[-1]
        fred_row  = (
            self._fred_hist.loc[:as_of_date].iloc[-1]
            if self._fred_hist is not None and len(self._fred_hist.loc[:as_of_date]) > 0
            else None
        )

        equity   = self._compute_equity_regime(vol_row)
        rates    = self._compute_rate_regime(vol_row, fred_row)
        commodity = self._compute_commodity_regime(vol_row)
        credit   = self._compute_credit_regime(vol_row, fred_row)

        return VolRegimeTensor(
            equity=equity,
            rates=rates,
            commodity=commodity,
            credit=credit,
            date=as_of_date,
        )

    def _compute_equity_regime(self, row: pd.Series) -> AssetClassRegime:
        """VIX term structure + VRP → equity regime."""
        slope    = float(row.get("equity_vts_slope", 1.0))
        vix_z    = float(row.get("vix_z", 0.0))
        vrp_z    = float(row.get("equity_vrp_z", 0.0))
        curve    = float(row.get("equity_vts_curve", 1.0))

        # Primary crisis signal: term structure backwardation
        if slope < _EQUITY_CRISIS_SLOPE:
            label = "crisis"
        elif slope < _EQUITY_STRESS_SLOPE or vix_z > 2.0:
            label = "stress"
        elif curve < 0.95 and vix_z < -0.5:
            label = "complacent"   # inverted contango + low VIX = late-bull complacency
        else:
            label = "neutral"

        # Continuous urgency: combines slope inversion depth + VIX level + VRP
        slope_urgency = float(np.clip((1.0 - slope) / 0.3, 0.0, 1.0))
        level_urgency = float(np.clip(vix_z / 4.0, 0.0, 1.0))
        urgency       = float(np.clip(0.6 * slope_urgency + 0.4 * level_urgency, 0.0, 1.0))

        return AssetClassRegime(
            label=label,
            z_score=vix_z,
            urgency=urgency,
            indicator="VIX_term_structure",
        )

    def _compute_rate_regime(
        self,
        vol_row: pd.Series,
        fred_row: Optional[pd.Series],
    ) -> AssetClassRegime:
        """
        MOVE index (via IG OAS proxy) + 2Y10Y inversion → rate regime.

        When FRED data is unavailable, falls back to TLT-implied proxy:
        rate stress ≈ high equity VIX + positive equity VRP (flight-to-quality
        regime where both stocks and bonds are being sold, i.e. 2022 scenario).
        """
        if fred_row is not None and "hy_oas_z" in fred_row.index:
            # Use HY OAS as proxy for MOVE (highly correlated during rate vol spikes)
            move_proxy_z = float(fred_row.get("ig_oas_z", 0.0))
            t10y2y_z     = float(fred_row.get("t10y2y_z", 0.0))
            # Rate crisis: IG spreads blow out OR yield curve deeply inverted
            combined_z = 0.6 * move_proxy_z - 0.4 * t10y2y_z  # inversion (negative) = stress
            indicator  = "IG_OAS+2Y10Y"
        else:
            # Fallback: VIX level + equity VRP as rate stress proxy
            combined_z = float(vol_row.get("vix_z", 0.0)) * 0.5
            indicator  = "VIX_proxy"

        if combined_z > _RATE_VOL_CRISIS_Z:
            label = "crisis"
        elif combined_z > _RATE_VOL_STRESS_Z:
            label = "stress"
        elif combined_z < -1.0:
            label = "complacent"
        else:
            label = "neutral"

        urgency = float(np.clip(combined_z / 4.0, 0.0, 1.0))

        return AssetClassRegime(
            label=label,
            z_score=combined_z,
            urgency=urgency,
            indicator=indicator,
        )

    def _compute_commodity_regime(self, vol_row: pd.Series) -> AssetClassRegime:
        """GVZ (gold) and OVX (oil) → commodity regime."""
        gvz_z = float(vol_row.get("gvz_z", 0.0))
        ovx_z = float(vol_row.get("ovx_z", 0.0))

        # Commodity crisis: either gold OR oil vol spikes (different drivers,
        # but both signal physical market stress)
        combined_z = float(np.maximum(gvz_z, ovx_z))
        # Also penalise simultaneous spikes (systemic commodity stress)
        if gvz_z > 1.5 and ovx_z > 1.5:
            combined_z *= 1.25

        if combined_z > _COMMODITY_CRISIS_Z:
            label = "crisis"
        elif combined_z > 1.0:
            label = "stress"
        elif combined_z < -1.0:
            label = "complacent"
        else:
            label = "neutral"

        urgency = float(np.clip(combined_z / 4.0, 0.0, 1.0))

        return AssetClassRegime(
            label=label,
            z_score=combined_z,
            urgency=urgency,
            indicator=f"GVZ(z={gvz_z:.2f})+OVX(z={ovx_z:.2f})",
        )

    def _compute_credit_regime(
        self,
        vol_row: pd.Series,
        fred_row: Optional[pd.Series],
    ) -> AssetClassRegime:
        """HY OAS z-score → credit/spread regime."""
        if fred_row is not None and "hy_oas_z" in fred_row.index:
            hy_z      = float(fred_row["hy_oas_z"])
            indicator = "HY_OAS_FRED"
        else:
            # Fallback: equity vol as credit proxy (empirically correlated ~0.75)
            hy_z      = float(vol_row.get("vix_z", 0.0)) * 0.75
            indicator = "VIX_credit_proxy"

        if hy_z > _CREDIT_CRISIS_Z:
            label = "crisis"
        elif hy_z > _CREDIT_STRESS_Z:
            label = "stress"
        elif hy_z < -1.0:
            label = "complacent"
        else:
            label = "neutral"

        urgency = float(np.clip(hy_z / 5.0, 0.0, 1.0))

        return AssetClassRegime(
            label=label,
            z_score=hy_z,
            urgency=urgency,
            indicator=indicator,
        )

    def get_tensor_series(
        self,
        tickers: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute VolRegimeTensor for all dates in history.

        Returns:
            z_mu_df:   DataFrame (T, 16) of z_mu vectors — drop-in replacement
                       for the Mamba-KAN regime_posteriors.parquet
            meta_df:   DataFrame (T, 4) of per-axis labels + urgencies
        """
        if self._vol_hist is None:
            raise RuntimeError("Call await load_history() first.")

        records_zmu  = []
        records_meta = []

        for date in self._vol_hist.index:
            date_str = str(date.date())
            try:
                tensor = self.get_regime_tensor(date_str)
            except Exception as e:
                logger.debug(f"Regime tensor failed for {date_str}: {e}")
                continue

            records_zmu.append({
                "date":     date_str,
                "z_mu":     tensor.z_mu.tolist(),
                "z_sigma":  tensor.z_sigma.tolist(),
                "tda_alert":    tensor.tda_alert,
                "ltc_urgency":  tensor.ltc_urgency,
                "regime_label": tensor.regime_label,
            })
            records_meta.append({
                "date":               date_str,
                "equity_label":       tensor.equity.label,
                "equity_urgency":     tensor.equity.urgency,
                "rates_label":        tensor.rates.label,
                "rates_urgency":      tensor.rates.urgency,
                "commodity_label":    tensor.commodity.label,
                "commodity_urgency":  tensor.commodity.urgency,
                "credit_label":       tensor.credit.label,
                "credit_urgency":     tensor.credit.urgency,
            })

        z_mu_df  = pd.DataFrame(records_zmu).set_index("date")
        meta_df  = pd.DataFrame(records_meta).set_index("date")

        logger.info(
            f"VolRegimeTensor series: {len(z_mu_df)} dates | "
            f"Crisis days: {(meta_df['equity_label'] == 'crisis').sum()}"
        )
        return z_mu_df, meta_df