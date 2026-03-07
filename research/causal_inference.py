"""
FORTRESS v5 - causal_inference.py
Path: research/causal_inference.py

Causal Graph Inference Engine for the GATv2 Alpha Engine.

Replaces the `build_dummy_edge_index()` scaffold. The GATv2 was learning
shock propagation on a random graph — a fatal flaw for any causal architecture.

IMPLEMENTATIONS:
  1. DYNOTEARS-Lite (Pamfil et al., 2020)
     Continuous-optimisation structural VAR with NOTEARS acyclicity constraint.
     Learns directed W[i,j] = "asset i Granger-causes asset j".
     Solver: Augmented Lagrangian Method (ALM) with L-BFGS-B inner loop.

  2. DCC Correlation Edges (Engle, 2002 — scalar DCC proxy)
     Rolling Pearson correlation. Assets with |ρ| > threshold get
     bidirectional edges. Encodes the `dcc_correlation` edge type.

  3. Macro Sensitivity Edges
     Assets sharing significant FRED factor loadings get connected.
     Populates the `macro_sensitivity` edge type.

LOOK-AHEAD SAFETY:
    `build()` enforces returns_df.index <= as_of_date before any computation.
    Raises LookAheadError on violation.

USAGE:
    from research.causal_inference import CausalGraphBuilder
    builder = CausalGraphBuilder(tickers=universe_tickers)
    edge_index, edge_attr = builder.build(returns_df, as_of_date='2024-06-01')
    graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger("CausalInference")


class LookAheadError(Exception):
    """Raised when returns data contains dates after as_of_date."""
    pass


# ── Hyperparameters ────────────────────────────────────────────────────────────
_DYNOTEARS_LAMBDA1:  float = 0.01    # L1 sparsity penalty on W
_DYNOTEARS_LAMBDA2:  float = 0.01    # Ridge (L2) regularisation on W
_DYNOTEARS_MAX_ITER: int   = 300     # Total ALM outer iterations
_DYNOTEARS_H_TOL:    float = 1e-8   # DAG constraint convergence tolerance
_DYNOTEARS_RHO_MAX:  float = 1e16   # Max augmented Lagrangian ρ

_DCC_CORR_THRESHOLD:  float = 0.55   # Min |ρ| for correlation edge
_DCC_ANTI_THRESHOLD:  float = -0.45  # Max ρ for inverse-correlation edge
_GRANGER_THRESHOLD:   float = 0.08   # Min |W[i,j]| to include a causal edge
_MACRO_SENS_THRESHOLD: float = 0.30  # Min |β| for macro-sensitivity edge

# Edge type indices — must match AssetGraph.EDGE_TYPES in gat_alpha.py
_EDGE_GRANGER:    int = 0
_EDGE_DCC:        int = 1
_EDGE_MACRO_SENS: int = 2
_EDGE_INST_FLOW:  int = 3
_EDGE_SUPPLY:     int = 4
_EDGE_FEAT_DIM:   int = 5


class CausalGraphBuilder:
    """
    Builds the multi-relational causal adjacency structure for GATv2.

    Args:
        tickers:       Ordered list of asset tickers matching node index 0..N-1.
        lookback_days: Rolling window for statistics (default 252 = 1 year).
        lag_order:     VAR lag order for DYNOTEARS (default 1 = next-day causality).
        macro_factors: Optional (T, K) DataFrame of macro factors for sensitivity edges.
    """

    def __init__(
        self,
        tickers: List[str],
        lookback_days: int = 252,
        lag_order: int = 1,
        macro_factors: Optional[pd.DataFrame] = None,
    ) -> None:
        self.tickers       = tickers
        self.n             = len(tickers)
        self.lookback_days = lookback_days
        self.lag_order     = lag_order
        self.macro_factors = macro_factors

    def build(
        self,
        returns_df: pd.DataFrame,
        as_of_date: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Constructs PyTorch Geometric edge tensors from a return matrix.

        Args:
            returns_df: Daily log-returns, shape (T, N_assets). Columns = tickers.
            as_of_date: ISO date string. Rows after this date raise LookAheadError.

        Returns:
            edge_index: LongTensor  (2, E)  — COO-format directed edge list.
            edge_attr:  FloatTensor (E, 5)  — multi-relational edge features.
        """
        # ── Look-ahead firewall ───────────────────────────────────────────────
        if as_of_date is not None:
            cutoff = pd.Timestamp(as_of_date)
            if not isinstance(returns_df.index, pd.DatetimeIndex):
                returns_df.index = pd.to_datetime(returns_df.index)
            future_rows = returns_df[returns_df.index > cutoff]
            if len(future_rows) > 0:
                raise LookAheadError(
                    f"returns_df contains {len(future_rows)} rows AFTER as_of_date="
                    f"'{as_of_date}'. LOOK-AHEAD BIAS DETECTED in causal graph."
                )

        # ── Trim to lookback window ───────────────────────────────────────────
        df = returns_df.tail(self.lookback_days).copy()
        if df.shape[0] < max(30, self.lag_order + 2):
            logger.warning(
                f"Only {df.shape[0]} rows available. "
                "Falling back to correlation-only graph."
            )
            return self._correlation_only_graph(df)

        # Align to known tickers
        available = [t for t in self.tickers if t in df.columns]
        if len(available) < 2:
            logger.error("Fewer than 2 tickers in returns_df. Cannot build graph.")
            return self._empty_graph()

        R = df[available].fillna(0.0).values.astype(np.float64)  # (T, N)
        N = len(available)

        edge_sources:  List[int]         = []
        edge_targets:  List[int]         = []
        edge_features: List[List[float]] = []

        # ── 1. DYNOTEARS-Lite: Granger causal edges ───────────────────────────
        try:
            W = self._dynotears_lite(R, self.lag_order)
            for i in range(N):
                for j in range(N):
                    if i != j and abs(W[i, j]) > _GRANGER_THRESHOLD:
                        edge_sources.append(i)
                        edge_targets.append(j)
                        feat = [0.0] * _EDGE_FEAT_DIM
                        feat[_EDGE_GRANGER] = float(W[i, j])  # Signed causal strength
                        edge_features.append(feat)
            logger.debug(
                f"DYNOTEARS: {int(np.sum(np.abs(W) > _GRANGER_THRESHOLD))} causal edges."
            )
        except Exception as exc:
            logger.warning(f"DYNOTEARS failed: {exc}. Skipping Granger edges.")

        # ── 2. DCC Correlation edges ──────────────────────────────────────────
        corr = np.corrcoef(R.T)  # (N, N) Pearson correlation matrix
        for i in range(N):
            for j in range(i + 1, N):
                rho = float(corr[i, j])
                if abs(rho) < min(abs(_DCC_CORR_THRESHOLD), abs(_DCC_ANTI_THRESHOLD)):
                    continue

                # Both directions for undirected correlation
                for src, tgt in [(i, j), (j, i)]:
                    edge_sources.append(src)
                    edge_targets.append(tgt)
                    feat = [0.0] * _EDGE_FEAT_DIM
                    # Positive ρ → positive DCC value; negative ρ → negative (hedge)
                    feat[_EDGE_DCC] = rho
                    edge_features.append(feat)

        # ── 3. Macro Sensitivity edges ────────────────────────────────────────
        if self.macro_factors is not None:
            macro_edges = self._build_macro_sensitivity_edges(R, available, N)
            for src, tgt, feat in macro_edges:
                edge_sources.append(src)
                edge_targets.append(tgt)
                edge_features.append(feat)

        # ── Assemble PyTorch tensors ──────────────────────────────────────────
        if not edge_sources:
            logger.warning("No edges constructed — returning empty graph.")
            return self._empty_graph()

        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr  = torch.tensor(edge_features, dtype=torch.float32)

        n_granger = int((edge_attr[:, _EDGE_GRANGER] != 0).sum())
        n_dcc     = int((edge_attr[:, _EDGE_DCC]     != 0).sum())
        n_macro   = int((edge_attr[:, _EDGE_MACRO_SENS] != 0).sum())

        logger.info(
            f"Causal graph: {N} nodes | {len(edge_sources)} edges "
            f"[Granger={n_granger} | DCC={n_dcc} | MacroSens={n_macro}]"
        )

        return edge_index, edge_attr

    # ── DYNOTEARS-Lite ────────────────────────────────────────────────────────

    def _dynotears_lite(self, R: np.ndarray, lag: int = 1) -> np.ndarray:
        """
        Continuous-optimisation structural VAR with NOTEARS acyclicity constraint.

        Reference: Pamfil et al., "DYNOTEARS: Structure Learning from Time-Series
        Data", AISTATS 2020.

        Formulation:
            min_{W}  (1/2T)||X_t - X_{t-lag} W||²_F
                     + λ1 ||W||_1 + (λ2/2) ||W||²_F
            s.t.     h(W) = tr(exp(W∘W)) - d = 0   [NOTEARS DAG constraint]

        Solved via Augmented Lagrangian Method (ALM):
            L(W, α, ρ) = obj(W) + α·h(W) + (ρ/2)·h(W)²
        with L-BFGS-B inner minimiser and dual ascent for α.

        Args:
            R:   Log-return matrix, shape (T, N).
            lag: VAR lag order (days).

        Returns:
            W: Causal weight matrix (N, N). W[i,j] = causal effect i→j.
        """
        T, N = R.shape

        # Build lagged data matrices
        X_curr = R[lag:, :]    # (T-lag, N) — targets
        X_lag  = R[:-lag, :]   # (T-lag, N) — predictors
        T_eff  = T - lag

        # Standardise each column for numerical conditioning
        eps = 1e-8
        X_curr = (X_curr - X_curr.mean(0)) / (X_curr.std(0) + eps)
        X_lag  = (X_lag  - X_lag.mean(0))  / (X_lag.std(0)  + eps)

        W = np.zeros((N, N), dtype=np.float64)

        # ── ALM state ────────────────────────────────────────────────────────
        rho     = 1.0
        alpha   = 0.0   # Lagrange multiplier for h(W) = 0

        def _h(W_: np.ndarray) -> float:
            """
            NOTEARS DAG constraint: h(W) = tr(exp(W∘W)) - N.
            Approximated via the matrix power series for efficiency:
                exp(A) ≈ (I + A/k)^k   for large k
            """
            A = W_ * W_ / N  # element-wise square, scaled
            # Use a 4th-order approximation: (I + A)^4 ~ exp(4A)
            M  = np.eye(N) + A
            M4 = M @ M @ M @ M
            return float(np.trace(M4)) - N

        def _grad_h_fd(W_: np.ndarray, eps_fd: float = 1e-5) -> np.ndarray:
            """Finite-difference gradient of h(W)."""
            h0   = _h(W_)
            grad = np.zeros_like(W_)
            for i in range(N):
                for j in range(N):
                    W_tmp      = W_.copy()
                    W_tmp[i,j] += eps_fd
                    grad[i,j]  = (_h(W_tmp) - h0) / eps_fd
            return grad

        def _objective(W_flat: np.ndarray) -> float:
            W_ = W_flat.reshape(N, N)
            residual = X_curr - X_lag @ W_
            mse     = 0.5 * float(np.sum(residual ** 2)) / T_eff
            l1      = _DYNOTEARS_LAMBDA1 * float(np.sum(np.abs(W_)))
            l2      = _DYNOTEARS_LAMBDA2 * 0.5 * float(np.sum(W_ ** 2))
            h_val   = _h(W_)
            penalty = 0.5 * rho * h_val ** 2 + alpha * h_val
            return mse + l1 + l2 + penalty

        def _gradient(W_flat: np.ndarray) -> np.ndarray:
            W_        = W_flat.reshape(N, N)
            residual  = X_curr - X_lag @ W_
            grad_mse  = -(X_lag.T @ residual) / T_eff
            grad_l1   = _DYNOTEARS_LAMBDA1 * np.sign(W_)
            grad_l2   = _DYNOTEARS_LAMBDA2 * W_
            h_val     = _h(W_)
            grad_h    = _grad_h_fd(W_)
            grad_pen  = (rho * h_val + alpha) * grad_h
            return (grad_mse + grad_l1 + grad_l2 + grad_pen).flatten()

        try:
            from scipy.optimize import minimize as _minimize
        except ImportError as exc:
            raise ImportError(
                "scipy is required for DYNOTEARS: pip install scipy"
            ) from exc

        # ── ALM outer loop ────────────────────────────────────────────────────
        for outer_iter in range(_DYNOTEARS_MAX_ITER // 10):
            result = _minimize(
                fun=_objective,
                x0=W.flatten(),
                jac=_gradient,
                method="L-BFGS-B",
                options={"maxiter": 50, "ftol": 1e-14, "gtol": 1e-8},
            )
            W = result.x.reshape(N, N)

            h_val = _h(W)
            alpha = alpha + rho * h_val     # Dual ascent
            rho   = min(rho * 10.0, _DYNOTEARS_RHO_MAX)

            if abs(h_val) < _DYNOTEARS_H_TOL:
                logger.debug(
                    f"DYNOTEARS converged at outer iter {outer_iter} "
                    f"(h={h_val:.2e})."
                )
                break

        # Post-process: zero diagonal + threshold small weights
        np.fill_diagonal(W, 0.0)
        W[np.abs(W) < _GRANGER_THRESHOLD] = 0.0

        return W

    # ── Macro sensitivity edges ───────────────────────────────────────────────

    def _build_macro_sensitivity_edges(
        self,
        R: np.ndarray,
        available: List[str],
        N: int,
    ) -> List[Tuple[int, int, List[float]]]:
        """
        Connects assets with shared sensitivity to common macro factors.

        Method: For each FRED factor column in self.macro_factors, regress
        each asset's return on the factor to get β_i. If |β_i - β_j| < threshold,
        the two assets share macro sensitivity → connect with a macro edge.

        Returns:
            List of (src, tgt, edge_feat) tuples.
        """
        edges = []
        if self.macro_factors is None or self.macro_factors.shape[1] == 0:
            return edges

        # Align factor and return dates
        common_idx = self.macro_factors.index.intersection(
            pd.RangeIndex(len(R))  # Placeholder — in prod, align on datetime index
        )
        if len(common_idx) < 10:
            return edges

        F = self.macro_factors.iloc[:len(R)].values  # (T, K) factors
        if F.shape[0] != R.shape[0]:
            return edges

        K = F.shape[1]
        betas = np.zeros((N, K))  # β matrix: asset × factor loadings

        for i in range(N):
            for k in range(K):
                cov  = np.cov(R[:, i], F[:, k])
                var  = max(np.var(F[:, k]), 1e-12)
                betas[i, k] = cov[0, 1] / var

        # Connect pairs with similar factor loadings
        for i in range(N):
            for j in range(i + 1, N):
                # Cosine similarity of beta vectors as sensitivity similarity
                b_i  = betas[i]
                b_j  = betas[j]
                norm = max(np.linalg.norm(b_i) * np.linalg.norm(b_j), 1e-12)
                sim  = float(np.dot(b_i, b_j) / norm)

                if sim > _MACRO_SENS_THRESHOLD:
                    for src, tgt in [(i, j), (j, i)]:
                        feat = [0.0] * _EDGE_FEAT_DIM
                        feat[_EDGE_MACRO_SENS] = sim
                        edges.append((src, tgt, feat))

        return edges

    # ── Fallback graphs ───────────────────────────────────────────────────────

    def _correlation_only_graph(
        self, df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Minimal graph from Pearson correlation when DYNOTEARS cannot run."""
        available = [t for t in self.tickers if t in df.columns]
        if len(available) < 2:
            return self._empty_graph()

        R = df[available].fillna(0.0).values.astype(np.float64)
        corr = np.corrcoef(R.T)
        N = len(available)

        sources, targets, feats = [], [], []
        for i in range(N):
            for j in range(i + 1, N):
                rho = float(corr[i, j])
                if abs(rho) >= _DCC_CORR_THRESHOLD:
                    for src, tgt in [(i, j), (j, i)]:
                        sources.append(src)
                        targets.append(tgt)
                        f = [0.0] * _EDGE_FEAT_DIM
                        f[_EDGE_DCC] = rho
                        feats.append(f)

        if not sources:
            return self._empty_graph()

        return (
            torch.tensor([sources, targets], dtype=torch.long),
            torch.tensor(feats, dtype=torch.float32),
        )

    def _empty_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns an empty graph (no edges) as a safe fallback."""
        return (
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0, _EDGE_FEAT_DIM), dtype=torch.float32),
        )

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_edge_statistics(
        self, edge_attr: torch.Tensor
    ) -> Dict[str, int]:
        """Returns a breakdown of edges by type for logging."""
        return {
            "granger":    int((edge_attr[:, _EDGE_GRANGER]    != 0).sum()),
            "dcc":        int((edge_attr[:, _EDGE_DCC]        != 0).sum()),
            "macro_sens": int((edge_attr[:, _EDGE_MACRO_SENS] != 0).sum()),
            "inst_flow":  int((edge_attr[:, _EDGE_INST_FLOW]  != 0).sum()),
            "supply":     int((edge_attr[:, _EDGE_SUPPLY]     != 0).sum()),
            "total":      edge_attr.shape[0],
        }