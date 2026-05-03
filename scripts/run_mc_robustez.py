"""
scripts/run_mc_robustez.py
──────────────────────────
Fortress v5 — Monte Carlo de Robustez (semillas aleatorias)

Reemplaza el def main() de run_v5_backtest.py para ejecutar el
walk-forward completo (9 folds) con múltiples semillas y medir
si los resultados son estables o dependen de la inicialización.

Uso:
  PYTHONPATH=. python scripts/run_mc_robustez.py

Salidas:
  research/outputs/mc_robustez_summary.csv   ← una fila por seed
  research/outputs/mc_robustez_folds.csv     ← detalle fold×seed
"""
from __future__ import annotations

import logging
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ── Importamos TODO del script original ────────────────────────────────────────
# Asegúrate de que PYTHONPATH=. esté activo al ejecutar.
from scripts.run_v5_backtest import (
    # clases y funciones del backtest
    V5Backtester,
    FoldResult,
    _load_data,
    _build_hmm_features,
    _sharpe,
    _sortino,
    _max_dd,
    _cagr,
    _calmar,
    _cost_drag_bps,
    # constantes
    _FOLD_SCHEDULE,
    _BASELINE_WF,
    _OUT_DIR,
    _PRICES_P,
    _RETURNS_P,
    _ALPHA_P,
    TICKERS,
    INITIAL_CAP,
    EMBARGO_DAYS,
    _MIN_PERIODS,
    _V5_AVAILABLE,
    RF_DAILY,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MC_Robustez")

# ── Configuración Monte Carlo ──────────────────────────────────────────────────
MC_SEEDS: List[int] = [42, 7, 123, 99, 555]

# Umbrales de pase (peor caso entre todas las seeds)
THRESH_CAGR_MIN: float  = 0.0      # CAGR promedio OOS > 0%
THRESH_MAXDD_MIN: float = -0.05    # MaxDD peor fold > -5%  (i.e. < 5% caída)

_MC_SUMMARY_CSV = _OUT_DIR / "mc_robustez_summary.csv"
_MC_FOLDS_CSV   = _OUT_DIR / "mc_robustez_folds.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de reporte
# ─────────────────────────────────────────────────────────────────────────────

def _print_seed_header(seed: int, idx: int, total: int) -> None:
    W = 90
    logger.info("\n" + "═" * W)
    logger.info(
        f"  SEED {seed:>4}   [{idx}/{total}]   "
        f"Iniciando walk-forward (9 folds)..."
    )
    logger.info("═" * W)


def _print_mc_summary(
    mc_summary: List[dict],
    thresholds: dict,
) -> None:
    W   = 112
    SEP = "─" * W
    logger.info("\n" + "═" * W)
    logger.info("  FORTRESS v5 — MONTE CARLO ROBUSTEZ")
    logger.info(
        "  Umbrales: avg CAGR > %.0f%%  |  peor MaxDD fold > %.0f%%",
        thresholds["cagr_min"] * 100,
        thresholds["maxdd_min"] * 100,
    )
    logger.info("═" * W)

    hdr = (
        f"  {'Seed':>5}  {'Folds':>5}  "
        f"{'avg CAGR':>10}  {'avg SR':>8}  {'avg Srt':>9}  "
        f"{'worst MDD':>10}  {'avg Turn':>9}  {'Cost bps':>9}  "
        f"{'std CAGR':>9}  {'Estado':>8}"
    )
    logger.info(hdr)
    logger.info(SEP)

    for r in mc_summary:
        flag  = "✓ PASS" if r["pass"] else "✗ FAIL"
        logger.info(
            f"  {r['seed']:>5}  {r['n_folds']:>5}  "
            f"  {r['avg_cagr']:>+8.2%}  {r['avg_sharpe']:>+8.3f}  "
            f"  {r['avg_sortino']:>+9.3f}  {r['worst_maxdd']:>+10.2%}  "
            f"  {r['avg_turnover']:>8.2%}  {r['avg_cost_bps']:>9.1f}  "
            f"  {'':>9}  {flag:>8}"
        )

    logger.info(SEP)

    all_cagr = [r["avg_cagr"]    for r in mc_summary]
    all_mdd  = [r["worst_maxdd"] for r in mc_summary]
    all_sr   = [r["avg_sharpe"]  for r in mc_summary]
    n_pass   = sum(1 for r in mc_summary if r["pass"])
    n_total  = len(mc_summary)

    cagr_std   = float(np.std(all_cagr)) * 100
    cagr_range = (max(all_cagr) - min(all_cagr)) * 100

    logger.info(
        f"\n  Estadísticas de estabilidad:"
        f"\n    CAGR avg  : {np.mean(all_cagr):+.2%}  |  "
        f"min {min(all_cagr):+.2%}  /  max {max(all_cagr):+.2%}"
        f"\n    CAGR σ    : {cagr_std:.2f}pp   (spread max−min: {cagr_range:.2f}pp)"
        f"\n    Peor MaxDD: {min(all_mdd):+.2%}  (peor fold de cualquier seed)"
        f"\n    Seeds PASS: {n_pass} / {n_total}"
    )

    # ── Veredicto final ───────────────────────────────────────────────────────
    logger.info("\n" + "═" * W)
    if n_pass == n_total:
        logger.info("  VEREDICTO → GO  ✓  Sistema robusto. Puede lanzarse a real.")
        logger.info(
            "  Todas las seeds superan los umbrales. "
            f"Peor caso: CAGR {min(all_cagr):+.2%} · MaxDD {min(all_mdd):+.2%}."
        )
    elif n_pass >= int(n_total * 0.8):
        logger.info("  VEREDICTO → GO con cautela ⚠  Mayoría de seeds pasan.")
        logger.info(
            "  Monitorizar de cerca en producción. "
            f"Seeds fallidas: {[r['seed'] for r in mc_summary if not r['pass']]}"
        )
    else:
        logger.info("  VEREDICTO → NO-GO  ✗  Sistema inestable.")
        logger.info(
            "  Demasiadas seeds fallan los umbrales. "
            "Considera simplificar el HMM o reducir la complejidad del modelo."
        )
    logger.info("═" * W)


# ─────────────────────────────────────────────────────────────────────────────
# main() — reemplaza el de run_v5_backtest.py
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("══════ Fortress v5 — Monte Carlo de Robustez ══════")
    logger.info("Seeds: %s", MC_SEEDS)
    logger.info("Umbrales: CAGR > %.0f%%  |  MaxDD > %.0f%%",
                THRESH_CAGR_MIN * 100, THRESH_MAXDD_MIN * 100)

    # ── Verificar archivos necesarios ─────────────────────────────────────────
    for p in [_PRICES_P, _RETURNS_P, _ALPHA_P]:
        if not p.exists():
            logger.error("Cache faltante: %s", p)
            logger.error("Ejecuta scripts/precompute_alpha_signals.py primero.")
            sys.exit(1)

    # ── Cargar datos UNA sola vez (caro — no repetir por seed) ────────────────
    prices_df, returns_df, regime_df, alpha_df = _load_data()
    all_dates = prices_df.index

    logger.info("Construyendo features HMM para todo el rango (%d días)...", len(all_dates))
    hmm_feats_all = _build_hmm_features(returns_df, prices_df, all_dates)

    # ── Monte Carlo: bucle sobre seeds ───────────────────────────────────────
    mc_results : Dict[int, List[FoldResult]] = {}
    mc_summary : List[dict]                  = []

    for run_idx, seed in enumerate(MC_SEEDS, start=1):
        _print_seed_header(seed, run_idx, len(MC_SEEDS))

        # ── Fijar todas las semillas aleatorias ───────────────────────────────
        np.random.seed(seed)
        random.seed(seed)
        if _V5_AVAILABLE:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # ── Instanciar backtester fresco para esta seed ───────────────────────
        backtester    = V5Backtester()
        fold_results  : List[FoldResult] = []
        all_oos_daily : List[float]      = []

        for fid, (is_s, is_e, oos_s, oos_e) in enumerate(_FOLD_SCHEDULE, start=1):

            # Slices IS/OOS
            is_mask  = (prices_df.index >= is_s) & (prices_df.index <= is_e)
            oos_mask = (prices_df.index >= oos_s) & (prices_df.index <= oos_e)

            # Embargo: quitar los últimos EMBARGO_DAYS del IS
            is_idx = np.where(is_mask)[0]
            if len(is_idx) > EMBARGO_DAYS:
                is_mask_clean = np.zeros(len(prices_df), dtype=bool)
                is_mask_clean[is_idx[:-EMBARGO_DAYS]] = True
            else:
                is_mask_clean = is_mask

            prices_is   = prices_df[is_mask_clean].reindex(columns=TICKERS).ffill()
            returns_is  = returns_df[is_mask_clean].reindex(columns=TICKERS).fillna(0.0)
            alpha_is    = alpha_df[is_mask_clean].reindex(columns=TICKERS).fillna(0.0)
            prices_oos  = prices_df[oos_mask].reindex(columns=TICKERS).ffill()
            returns_oos = returns_df[oos_mask].reindex(columns=TICKERS).fillna(0.0)
            alpha_oos   = alpha_df[oos_mask].reindex(columns=TICKERS).fillna(0.0)

            if len(prices_is) < _MIN_PERIODS or len(prices_oos) < 5:
                logger.warning("  [seed %d] F%d: datos insuficientes — skip", seed, fid)
                continue

            hmm_feats_is = hmm_feats_all[is_mask_clean]

            # IS: entrenar HMM + calibrar conformal
            backtester._init_fold(
                prices_is    = prices_is,
                returns_is   = returns_is,
                alpha_is     = alpha_is,
                hmm_feats_is = hmm_feats_is,
            )

            # OOS: simular
            oos_dates = prices_oos.index
            daily_ret, daily_turn, exposures = backtester.run_oos(
                prices_oos    = prices_oos,
                returns_oos   = returns_oos,
                alpha_oos     = alpha_oos,
                hmm_feats_all = hmm_feats_all,
                all_dates     = all_dates,
                oos_dates     = oos_dates,
            )

            if len(daily_ret) < 5:
                logger.warning("  [seed %d] F%d: OOS devuelve < 5 días", seed, fid)
                continue

            r_arr   = np.array(daily_ret)
            t_arr   = np.array(daily_turn)
            e_arr   = np.array(exposures)
            nav_arr = INITIAL_CAP * np.cumprod(1.0 + r_arr)

            fr = FoldResult(
                fold_id      = fid,
                is_start     = is_s,   is_end     = is_e,
                oos_start    = oos_s,  oos_end    = oos_e,
                oos_sharpe   = round(_sharpe(r_arr),   4),
                oos_sortino  = round(_sortino(r_arr),  4),
                oos_max_dd   = round(_max_dd(nav_arr), 5),
                oos_cagr     = round(_cagr(nav_arr, len(r_arr)), 4),
                oos_calmar   = round(_calmar(r_arr, nav_arr), 4),
                avg_turnover = round(float(t_arr.mean()), 5),
                cost_drag_bps= round(_cost_drag_bps(t_arr), 2),
                avg_exposure = round(float(e_arr.mean()), 4),
                masked_pct   = round(float((e_arr < 0.95).mean() * 100), 1),
                daily_returns= [round(x, 8) for x in daily_ret],
            )
            fold_results.append(fr)
            all_oos_daily.extend(daily_ret)

            logger.info(
                "  [seed %d] F%d → SR=%+.3f  CAGR=%+.2f%%  MaxDD=%.2f%%",
                seed, fid,
                fr.oos_sharpe,
                fr.oos_cagr * 100,
                fr.oos_max_dd * 100,
            )

        # ── Métricas agregadas de esta seed ───────────────────────────────────
        mc_results[seed] = fold_results

        if not fold_results:
            logger.warning("  [seed %d] Sin folds válidos.", seed)
            continue

        avg_cagr   = float(np.mean([r.oos_cagr    for r in fold_results]))
        avg_sr     = float(np.mean([r.oos_sharpe   for r in fold_results]))
        avg_srt    = float(np.mean([r.oos_sortino  for r in fold_results]))
        worst_mdd  = float(min(r.oos_max_dd         for r in fold_results))
        avg_turn   = float(np.mean([r.avg_turnover  for r in fold_results]))
        avg_cost   = float(np.mean([r.cost_drag_bps for r in fold_results]))

        # CAGR across-fold std (estabilidad intrafold)
        fold_cagr_std = float(np.std([r.oos_cagr for r in fold_results]))

        passed = avg_cagr > THRESH_CAGR_MIN and worst_mdd > THRESH_MAXDD_MIN

        mc_summary.append({
            "seed"         : seed,
            "n_folds"      : len(fold_results),
            "avg_cagr"     : round(avg_cagr,  4),
            "avg_sharpe"   : round(avg_sr,    4),
            "avg_sortino"  : round(avg_srt,   4),
            "worst_maxdd"  : round(worst_mdd, 5),
            "avg_turnover" : round(avg_turn,  5),
            "avg_cost_bps" : round(avg_cost,  2),
            "fold_cagr_std": round(fold_cagr_std, 4),
            "pass"         : passed,
        })

        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(
            "  ▶ SEED %d %s  avg CAGR: %+.2f%%  worst MaxDD: %.2f%%  avg SR: %.3f",
            seed, status, avg_cagr * 100, worst_mdd * 100, avg_sr,
        )

    if not mc_summary:
        logger.error("No se completó ninguna seed. Saliendo.")
        sys.exit(1)

    # ── Imprimir resumen Monte Carlo ──────────────────────────────────────────
    _print_mc_summary(
        mc_summary,
        {"cagr_min": THRESH_CAGR_MIN, "maxdd_min": THRESH_MAXDD_MIN},
    )

    # ── Guardar CSVs ──────────────────────────────────────────────────────────
    pd.DataFrame(mc_summary).to_csv(_MC_SUMMARY_CSV, index=False)
    logger.info("✅ Resumen MC → %s", _MC_SUMMARY_CSV)

    all_rows = []
    for seed, results in mc_results.items():
        for r in results:
            row = {k: v for k, v in asdict(r).items() if k != "daily_returns"}
            row["seed"] = seed
            all_rows.append(row)
    pd.DataFrame(all_rows).to_csv(_MC_FOLDS_CSV, index=False)
    logger.info("✅ Detalle folds MC → %s", _MC_FOLDS_CSV)

    logger.info("\n══ FIN DEL MONTE CARLO DE ROBUSTEZ ══")


if __name__ == "__main__":
    main()