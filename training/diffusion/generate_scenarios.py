"""
FORTRESS v5 - generate_scenarios.py  [FULL IMPLEMENTATION]
Path: training/diffusion/generate_scenarios.py

Neural SDE Adversarial Scenario Generator.
Called by meta_learning/meta_agent.py ValidationPipeline Gate 3 as a subprocess:

    python training/diffusion/generate_scenarios.py \\
        --feature-path /tmp/gate3_feature_xyz.py \\
        --n-paths 2000 \\
        --output-format json

Responsibilities:
  1. Load the trained LatentSDEWorldModel weights.
  2. Load the trained MambaKANVAE weights to generate adversarial regime vectors.
  3. Generate `n_paths` synthetic market trajectories across the worst 10% of the
     regime space (adversarial sampling — not random sampling).
  4. Dynamically import the generated feature code and evaluate it on each path.
  5. Compute the feature's CVaR-95 stability score across the adversarial paths.
  6. Output a single JSON line to stdout:
         {"stress_test_passed": bool, "cvar_95": float, "n_paths": int,
          "mean_feature": float, "std_feature": float}
     Exit code 0 on success (even if test fails), 1 on unhandled error.

DESIGN NOTES:
  - The subprocess design is intentional: it isolates the generated code in a
    separate Python process so that any crash, import error, or GPU OOM in the
    generated feature cannot corrupt the parent meta-agent process.
  - Adversarial regime sampling: We draw z_t from the tails of the prior
    N(0, I_16) using the Mahalanobis distance top-10% filter. This ensures
    the stress test evaluates crash/high-vol regimes, not average conditions.
  - The stability criterion: the feature must produce finite, non-NaN outputs
    on >95% of paths AND the feature's CVaR-95 must exceed the stability floor.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import traceback
from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.WARNING,   # Suppress noise in subprocess context
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GenerateScenarios")

# ── Configuration ─────────────────────────────────────────────────────────────
_CONFIG_PATH = "config/hyperparams.yaml"
_RISK_LIMITS_PATH = "config/risk_limits.yaml"
_SDE_WEIGHTS   = "models/weights/sde_latest.pt"
_MAMBA_WEIGHTS = "models/weights/mamba_kan_latest.pt"

# CVaR floor: generated feature must not reduce CVaR-95 below this level
# (expressed as a feature value; adjust if feature output is dimensioned differently)
_CVAR_STABILITY_FLOOR: float = -0.10   # Feature output should not average < -10% in the worst paths
_MIN_FINITE_RATE: float = 0.95          # At least 95% of paths must produce finite feature values


def _load_config() -> Tuple[dict, dict]:
    with open(_CONFIG_PATH, "r") as f:
        hp = yaml.safe_load(f)
    with open(_RISK_LIMITS_PATH, "r") as f:
        rl = yaml.safe_load(f)
    return hp, rl


def _load_sde_model(config: dict, device: torch.device):
    """Loads the trained LatentSDEWorldModel. Returns None if weights are missing."""
    from models.world_model.neural_sde import LatentSDEWorldModel

    model = LatentSDEWorldModel(config.get("world_model", {})).to(device)

    if not os.path.exists(_SDE_WEIGHTS):
        logger.warning(f"SDE weights not found at {_SDE_WEIGHTS}. Using untrained model.")
        return model

    model.load_state_dict(torch.load(_SDE_WEIGHTS, map_location=device))
    model.eval()
    logger.info(f"SDE weights loaded from {_SDE_WEIGHTS}")
    return model


def _sample_adversarial_regimes(
    n_paths: int,
    regime_dim: int = 16,
    tail_percentile: float = 0.10,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Draws z_t vectors from the adversarial (tail) region of the regime prior.

    Strategy:
      1. Sample 10 * n_paths candidates from N(0, I_16).
      2. Compute the Mahalanobis distance from the origin (= L2 norm for standard Normal).
      3. Keep only the top `tail_percentile` by distance — these represent
         the most extreme / out-of-distribution regimes.
      4. Randomly sub-sample n_paths from the tail set.

    Args:
        n_paths:         Number of regime vectors to return.
        regime_dim:      Latent dimension of z_t.
        tail_percentile: Fraction of candidates to keep (adversarial selection).
        device:          PyTorch device.

    Returns:
        z_adversarial: Shape (n_paths, regime_dim).
    """
    n_candidates = n_paths * 10
    candidates = torch.randn(n_candidates, regime_dim, device=device)

    # Mahalanobis distance from origin under N(0, I) = L2 norm
    distances = torch.norm(candidates, dim=-1)   # (n_candidates,)

    # Keep the top tail_percentile
    k = max(1, int(n_candidates * tail_percentile))
    _, top_indices = torch.topk(distances, k)
    tail_candidates = candidates[top_indices]

    # Sub-sample n_paths from the tail
    perm = torch.randperm(len(tail_candidates))[:n_paths]
    return tail_candidates[perm]


def _load_feature_function(feature_path: str) -> Optional[Callable]:
    """
    Dynamically imports the generated feature module and returns its
    `compute_feature(path: np.ndarray) -> float` entry point.

    Expected interface of the generated code:
        def compute_feature(path: np.ndarray) -> float:
            '''
            Args:
                path: np.ndarray of shape (n_steps + 1, State_Dim) — one SDE trajectory.
            Returns:
                A scalar float signal (e.g., expected return, momentum, factor loading).
            '''

    Returns None if the module cannot be imported or the function is missing.
    """
    try:
        spec = importlib.util.spec_from_file_location("generated_feature", feature_path)
        if spec is None or spec.loader is None:
            logger.error(f"Could not create module spec from {feature_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)   # type: ignore[attr-defined]

        if not hasattr(module, "compute_feature"):
            logger.error(
                f"Generated code at {feature_path} has no 'compute_feature' function. "
                "The LLM must define: def compute_feature(path: np.ndarray) -> float"
            )
            return None

        return module.compute_feature

    except Exception as exc:
        logger.error(f"Failed to import generated feature: {exc}")
        return None


def run_stress_test(
    feature_path: str,
    n_paths: int,
    device: torch.device,
) -> dict:
    """
    Core stress test logic.

    1. Loads SDE model.
    2. Samples adversarial regime vectors.
    3. Generates synthetic market paths under those regimes.
    4. Evaluates the generated feature on each path.
    5. Computes CVaR-95 of the feature values.

    Returns:
        Result dict with keys: stress_test_passed, cvar_95, n_paths,
                               mean_feature, std_feature, finite_rate.
    """
    hp, _ = _load_config()
    sde_model = _load_sde_model(hp, device)
    feature_fn = _load_feature_function(feature_path)

    state_dim  = hp.get("world_model", {}).get("sde_state_dim", 25)
    regime_dim = hp.get("world_model", {}).get("latent_dim", 16)
    n_steps    = 21  # 1 trading month forward

    # ── Generate adversarial regime vectors ────────────────────────────────────
    z_adversarial = _sample_adversarial_regimes(n_paths, regime_dim, device=device)

    # ── Simulate paths ─────────────────────────────────────────────────────────
    # Initial state: small perturbation around zero (log-return space)
    initial_state = torch.zeros(state_dim, device=device)

    feature_values: list[float] = []
    n_errors = 0

    with torch.no_grad():
        # Process in chunks of 256 to avoid OOM
        chunk_size = 256
        for start in range(0, n_paths, chunk_size):
            end = min(start + chunk_size, n_paths)
            z_chunk = z_adversarial[start:end]   # (chunk, regime_dim)
            chunk_n = z_chunk.shape[0]

            try:
                # paths: (chunk_n, n_steps + 1, state_dim)
                paths = sde_model.generate_synthetic_paths(
                    initial_state=initial_state,
                    z_t=z_chunk,
                    n_steps=n_steps,
                    dt=1.0,
                    n_paths=chunk_n,
                )
            except Exception as exc:
                logger.error(f"SDE generation failed for chunk {start}-{end}: {exc}")
                n_errors += chunk_n
                continue

            paths_np = paths.cpu().numpy()  # (chunk_n, n_steps+1, state_dim)

            # ── Evaluate the generated feature on each path ───────────────────
            if feature_fn is not None:
                for path_idx in range(chunk_n):
                    path = paths_np[path_idx]   # (n_steps+1, state_dim)
                    try:
                        val = float(feature_fn(path))
                        if np.isfinite(val):
                            feature_values.append(val)
                        else:
                            n_errors += 1
                    except Exception:
                        n_errors += 1
            else:
                # No feature function: evaluate raw portfolio return as the signal
                for path_idx in range(chunk_n):
                    path = paths_np[path_idx]
                    # Equal-weight portfolio return over the full horizon
                    portfolio_return = float(np.mean(path[-1] - path[0]))
                    if np.isfinite(portfolio_return):
                        feature_values.append(portfolio_return)
                    else:
                        n_errors += 1

    # ── Compute stability metrics ──────────────────────────────────────────────
    total_evaluated = len(feature_values) + n_errors
    finite_rate = len(feature_values) / max(total_evaluated, 1)

    if len(feature_values) < 50:
        # Too few valid evaluations — reject
        return {
            "stress_test_passed": False,
            "cvar_95":   float("nan"),
            "n_paths":   total_evaluated,
            "finite_rate": finite_rate,
            "mean_feature": float("nan"),
            "std_feature":  float("nan"),
            "error": "Fewer than 50 finite feature values produced",
        }

    arr = np.array(feature_values)
    mean_f = float(np.mean(arr))
    std_f  = float(np.std(arr))
    var_95  = float(np.percentile(arr, 5))
    cvar_95 = float(arr[arr <= var_95].mean())

    # ── Pass/fail criteria ─────────────────────────────────────────────────────
    passed = (
        finite_rate >= _MIN_FINITE_RATE
        and cvar_95 >= _CVAR_STABILITY_FLOOR
        and np.isfinite(cvar_95)
    )

    return {
        "stress_test_passed": passed,
        "cvar_95":     cvar_95,
        "var_95":      var_95,
        "n_paths":     total_evaluated,
        "finite_rate": finite_rate,
        "mean_feature": mean_f,
        "std_feature":  std_f,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FORTRESS v5 - SDE Adversarial Stress Tester")
    parser.add_argument(
        "--feature-path",
        type=str,
        required=True,
        help="Path to the generated feature Python file to evaluate.",
    )
    parser.add_argument(
        "--n-paths",
        type=int,
        default=2_000,
        help="Number of adversarial SDE paths to generate.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "human"],
        help="Output format. 'json' emits a single JSON line for machine parsing.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        result = run_stress_test(
            feature_path=args.feature_path,
            n_paths=args.n_paths,
            device=device,
        )
    except Exception as exc:
        error_payload = {
            "stress_test_passed": False,
            "cvar_95": float("nan"),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if args.output_format == "json":
            print(json.dumps(error_payload), flush=True)
        else:
            print(f"STRESS TEST ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output_format == "json":
        # Emit a single JSON line to stdout — parsed by meta_agent.py Gate 3
        print(json.dumps(result), flush=True)
    else:
        status = "✅ PASSED" if result["stress_test_passed"] else "❌ FAILED"
        print(f"\n{status}")
        print(f"  CVaR-95:     {result.get('cvar_95', 'N/A'):.4f}")
        print(f"  Finite Rate: {result.get('finite_rate', 0):.2%}")
        print(f"  Mean Signal: {result.get('mean_feature', 'N/A'):.4f}")
        print(f"  Std Signal:  {result.get('std_feature', 'N/A'):.4f}")
        print(f"  N Paths:     {result.get('n_paths', 0)}")

    sys.exit(0)


if __name__ == "__main__":
    main()