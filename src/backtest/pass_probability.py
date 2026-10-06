"""Monte Carlo de probabilidad de aprobar una evaluación con trailing DD (peak_unrealized)."""
import numpy as np


def arrays_from_trades(trades) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.array([t.r_multiple for t in trades], dtype=float)
    mfe = np.array([max(t.mfe_r, 0.0) for t in trades], dtype=float)
    mae = np.array([max(t.mae_r, 0.0) for t in trades], dtype=float)
    return r, mfe, mae


def pass_probability(
    r: np.ndarray, mfe: np.ndarray, mae: np.ndarray, *,
    risk_fraction: float,              # fracción del buffer arriesgada por trade (p.ej. 0.07)
    risk_cap: float = 250.0,
    edge_haircut: float = 0.0,         # resta a cada R (estrés: 0.079 => edge cero)
    initial: float = 50_000.0, dd: float = 2_000.0, target: float = 3_000.0,
    lock_offset: float = 100.0, trades_per_day: float = 0.49,
    n_sims: int = 20_000, max_trades: int = 800, seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n_sims, max_trades))
    equity = np.full(n_sims, initial)
    hwm = equity.copy()
    alive = np.ones(n_sims, bool)
    passed = np.zeros(n_sims, bool)
    breached = np.zeros(n_sims, bool)
    n_to_pass = np.full(n_sims, np.nan)

    for t in range(max_trades):
        floor = np.minimum(hwm - dd, initial + lock_offset)
        risk = np.minimum(risk_fraction * np.maximum(equity - floor, 0.0), risk_cap)
        i = idx[:, t]
        pnl = (r[i] - edge_haircut) * risk
        best = np.maximum(mfe[i] * risk, pnl)
        worst = np.minimum(-mae[i] * risk, pnl)

        new_hwm = np.where(alive, np.maximum(hwm, equity + best), hwm)
        new_floor = np.minimum(new_hwm - dd, initial + lock_offset)
        breach_now = alive & (equity + worst <= new_floor)

        hwm = new_hwm
        equity = np.where(alive, equity + pnl, equity)
        breached |= breach_now
        alive &= ~breach_now

        done = alive & (equity - initial >= target)
        n_to_pass[done] = t + 1
        passed |= done
        alive &= ~done
        if not alive.any():
            break

    med = np.nanmedian(n_to_pass) if passed.any() else float("nan")
    return {
        "p_pass": float(passed.mean()),
        "p_breach": float(breached.mean()),
        "p_unresolved": float((~passed & ~breached).mean()),
        "median_trades_to_pass": float(med),
        "median_days_to_pass": float(med / trades_per_day) if passed.any() else float("nan"),
    }