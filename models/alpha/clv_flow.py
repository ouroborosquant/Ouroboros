"""
FORTRESS v5 — models/alpha/clv_flow.py  [v1.0 — CLV-Flow]

Intraday Close Location Value Flow (CLV-Flow)
=============================================

Economic Thesis
---------------
In mega-cap equities and high-AUM ETFs, TWAP algorithms front-load execution
early in the session but must complete their daily quota before the closing
auction. The result is a systematic pattern in *where* the close occurs within
the intraday range, independent of whether the day's return is positive.

A stock that closes at the absolute high of its range (+1.0 CLV) on 3× normal
volume reveals aggressive algo buying into the close — the "Marking the Close"
phenomenon. Standard volume indicators (OBV, CMF) only compare close-to-close
direction; CLV-Flow captures the *intraday location* of institutional finish.

CLV Derivation
--------------
CLV_t = [(C_t − L_t) − (H_t − C_t)] / (H_t − L_t)
      = [2·C_t − H_t − L_t] / (H_t − L_t)

This maps the close's position in [L, H] linearly onto [−1, +1]:
  • CLV = +1.0: close == high  (maximum buying pressure)
  • CLV =  0.0: close == mid   (indeterminate)
  • CLV = −1.0: close == low   (maximum selling pressure)
  • H == L:     CLV = 0.0      (doji / halted; explicitly handled)

Flow Accumulation
-----------------
Flow_t = CLV_t × V_t   — capital-weighted intraday pressure

NetFlow_t = EMA_21(Flow_t) / SMA_21(V_t)
  — EMA smoothing extracts the persistent bias; volume normalisation
    converts raw share-count flow into a dimensionless fraction that is
    comparable across tickers (ETFs vs. equities have very different
    volume scales).

Signal = tanh( (NetFlow_t − μ_252) / (σ_252 · 3.0) )

Edge Case Handling
------------------
• H == L (doji, trading halt): CLV = 0.0 (no information, no signal contribution)
• Zero volume: forward-filled before EMA to prevent EMA decay to 0 from masked NaN
• Negative volume (data error): clipped to 0 before entry
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger("Ouroboros.CLVFlow")

_EMA_WINDOW:     int   = 21
_ZSCORE_WINDOW:  int   = 252
_TANH_SCALE:     float = 3.0
_MIN_VOL:        float = 1.0
_MIN_SIGMA:      float = 1e-6


class CLVFlowEngine:
    """
    Close Location Value Flow signal engine.

    Parameters
    ----------
    tickers : List[str]
        Universe in canonical order.
    ema_window : int
        EMA span for flow smoothing (and matching SMA for volume normalisation).
    zscore_window : int
        Trailing window for cross-time Z-score.
    """

    def __init__(
        self,
        tickers:       List[str],
        ema_window:    int   = _EMA_WINDOW,
        zscore_window: int   = _ZSCORE_WINDOW,
        tanh_scale:    float = _TANH_SCALE,
    ) -> None:
        self.tickers       = tickers
        self.ema_window    = ema_window
        self.zscore_window = zscore_window
        self.tanh_scale    = tanh_scale

    def compute_signal(
        self,
        high_df:   pd.DataFrame,
        low_df:    pd.DataFrame,
        close_df:  pd.DataFrame,
        volume_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        high_df / low_df / close_df : (T, N) daily OHLC.
        volume_df : (T, N) daily volume; must share the date index.

        Returns
        -------
        pd.DataFrame  (T, N)  ∈ (−1, 1)
        """
        H = high_df.reindex(columns=self.tickers).astype(np.float64)
        L = low_df.reindex(columns=self.tickers).astype(np.float64)
        C = close_df.reindex(columns=self.tickers).astype(np.float64)
        V = volume_df.reindex(columns=self.tickers).astype(np.float64).clip(lower=0.0)

        # ── Guard: yfinance adjusted data can give H < C or L > C ────────────
        H = np.maximum(H, C)
        L = np.minimum(L, C)

        # ── CLV: maps close location in [L, H] to [−1, +1] ──────────────────
        range_ = (H - L).clip(lower=0.0)
        # Where H == L (doji / halt), CLV is undefined → assign 0 explicitly
        clv = np.where(
            range_ < 1e-10,
            0.0,
            (2.0 * C - H - L) / range_,
        )
        clv = pd.DataFrame(clv, index=C.index, columns=self.tickers).clip(-1.0, 1.0)

        # ── Raw flow: CLV × volume (capital-weighted intraday pressure) ──────
        V_clean = V.replace(0.0, np.nan).ffill().fillna(_MIN_VOL).clip(lower=_MIN_VOL)
        flow    = clv * V_clean     # (T, N); units: shares × dimensionless scalar

        # ── EMA-smooth the flow series ───────────────────────────────────────
        flow_ema = flow.ewm(
            span=self.ema_window,
            adjust=False,
            min_periods=max(3, self.ema_window // 7),
        ).mean()

        # ── Normalise by rolling mean volume → dimensionless NetFlow ─────────
        # SMA_21(V) is the correct normaliser because EMA_21(Flow) uses the
        # same 21-bar span, so NetFlow ~= weighted_avg_CLV over the window.
        vol_sma = V_clean.rolling(self.ema_window, min_periods=max(3, self.ema_window // 7)).mean()
        net_flow = flow_ema / vol_sma.clip(lower=_MIN_VOL)   # (T, N)  ∈ [−1, +1]

        # ── Cross-time Z-score + tanh ─────────────────────────────────────────
        roll = net_flow.rolling(
            self.zscore_window, min_periods=max(63, self.zscore_window // 4)
        )
        mu  = roll.mean()
        sig = roll.std().clip(lower=_MIN_SIGMA)

        z      = (net_flow - mu) / (sig * self.tanh_scale)
        signal = np.tanh(z).astype(np.float32)

        result = pd.DataFrame(signal, index=C.index, columns=self.tickers)

        logger.info(
            f"CLV-Flow  | shape={result.shape} | "
            f"mean|α|={result.abs().mean().mean():.4f} | "
            f"non-zero={(result.abs() > 0.01).any(axis=1).mean()*100:.1f}%"
        )
        return result