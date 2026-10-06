import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
import yaml

from src.alpha.orb_breakout import ORBBreakoutStrategy
from src.alpha.trend_continuation import ORBTrendContinuationStrategy
from src.backtest.engine import BacktestEngine
from src.backtest.fee_models import CMEFeeModel
from src.core.events import BarEvent
from src.models.meta_classifier import MetaClassifier
from src.risk.position_sizer import InstrumentSpecs


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecutor de Backtests CME ORB")
    parser.add_argument("--data", type=str, default="data/raw/minute/cme_mnq_1m.parquet")
    parser.add_argument("--instrument", type=str, default="MNQ")
    parser.add_argument("--strategy-type", type=str, choices=["breakout", "retest"], default="retest", help="Tipo de estrategia")
    parser.add_argument("--prop-config", type=str, default="configs/prop_firm/evaluation_50k.yaml")
    parser.add_argument("--strategy-config", type=str, default="configs/strategy/orb_base.yaml")
    parser.add_argument("--meta-config", type=str, default="configs/strategy/meta_model.yaml")
    parser.add_argument("--weights", type=str, default="src/models/weights/lgbm_orb_meta_v1.booster")
    parser.add_argument("--cutoff", type=float, default=None, help="Sobrescribir cutoff de probabilidad")
    parser.add_argument("--no-meta", action="store_true", help="Desactivar filtro de ML y evaluar estrategia base")
    args = parser.parse_args()

    console = Console()

    with open(args.prop_config) as f:
        prop_cfg = yaml.safe_load(f)
    with open(args.strategy_config) as f:
        strat_cfg = yaml.safe_load(f)
    with open(args.meta_config) as f:
        meta_cfg = yaml.safe_load(f)

    cutoff = args.cutoff if args.cutoff is not None else meta_cfg["meta_labeling"].get("execution_probability_cutoff", 0.35)

    specs = InstrumentSpecs(
        symbol=args.instrument,
        tick_size=0.25,
        point_value=2.00,
        max_contracts=prop_cfg["limits"]["max_open_contracts"].get(args.instrument, 6),
    )

    fee_model = CMEFeeModel(commission_per_contract=0.62, slippage_ticks=1.0)

    if args.strategy_type == "retest":
        console.print("[bold cyan][*] Estrategia activa: Retesteo + Pullback a VWAP (Trend Continuation)[/bold cyan]")
        strategy = ORBTrendContinuationStrategy(
            symbol=args.instrument,
            buffer_ticks=strat_cfg["opening_range"]["buffer_ticks"],
            retest_tolerance_ticks=12,
            min_rvol_threshold=strat_cfg["entry_filters"]["min_rvol_threshold"],
            stop_atr_multiplier=1.25,
            risk_reward_ratio=strat_cfg["exit_rules"]["take_profit"]["risk_reward_ratio"],
            hard_stop_max_ticks=strat_cfg["exit_rules"]["stop_loss"]["hard_stop_max_ticks"],
        )
    else:
        console.print("[bold cyan][*] Estrategia activa: Ruptura Directa del Rango (Breakout)[/bold cyan]")
        strategy = ORBBreakoutStrategy(
            symbol=args.instrument,
            buffer_ticks=strat_cfg["opening_range"]["buffer_ticks"],
            min_rvol_threshold=strat_cfg["entry_filters"]["min_rvol_threshold"],
            min_range_atr_ratio=strat_cfg["entry_filters"]["min_range_atr_ratio"],
            max_range_atr_ratio=strat_cfg["entry_filters"]["max_range_atr_ratio"],
            risk_reward_ratio=strat_cfg["exit_rules"]["take_profit"]["risk_reward_ratio"],
            atr_multiplier=strat_cfg["exit_rules"]["stop_loss"]["atr_multiplier"],
            hard_stop_max_ticks=strat_cfg["exit_rules"]["stop_loss"]["hard_stop_max_ticks"],
        )

    meta_classifier = None
    weights_path = Path(args.weights)
    if not args.no_meta and weights_path.exists():
        console.print(f"[bold green][*] Cargando Meta-Clasificador (Cutoff={cutoff:.2f}) desde {weights_path}...[/bold green]")
        meta_classifier = MetaClassifier(probability_cutoff=cutoff)
        meta_classifier.load(weights_path)
    else:
        console.print("[yellow][*] Modo de ejecución: Estrategia SIN filtro secundario de ML.[/yellow]")

    engine = BacktestEngine(
        strategy=strategy,
        specs=specs,
        fee_model=fee_model,
        initial_capital=prop_cfg["initial_balance"],
        max_trailing_drawdown=prop_cfg["limits"]["max_trailing_drawdown"],
        max_daily_loss=prop_cfg["limits"]["max_daily_loss_limit"],
        meta_classifier=meta_classifier,
        debug=False,
    )

    console.print(f"[bold blue][*] Leyendo datos históricos desde {args.data}...[/bold blue]")
    df_raw = pd.read_parquet(args.data)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

    bars = [
        BarEvent(
            timestamp=row["timestamp"],
            symbol=row["symbol"],
            timeframe="1m",
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=float(row["volume"]),
        )
        for _, row in df_raw.iterrows()
    ]

    console.print(f"[bold blue][*] Ejecutando backtest sobre {len(bars):,} barras...[/bold blue]")
    result = engine.run(bars)
    m = result.metrics

    table = Table(title=f"📊 Resumen Estadístico ({args.strategy_type.upper()})")
    table.add_column("Métrica Cuantitativa", style="cyan", no_wrap=True)
    table.add_column("Valor Obtenido", style="bold green")

    table.add_row("Total de Operaciones", str(m.total_trades))
    table.add_row("Win Rate", f"{m.win_rate * 100:.2f}%")
    table.add_row("Profit Factor", f"{m.profit_factor:.2f}")
    table.add_row("Expectancy por Trade", f"${m.expectancy_dollars:,.2f}")
    table.add_row("Beneficio Neto (Net PnL)", f"${m.net_pnl:,.2f}")
    table.add_row("Comisiones + Slippage", f"${m.total_commissions + m.total_slippage:,.2f}")
    table.add_row("Sharpe Ratio (Anualizado)", f"{m.sharpe_ratio:.2f}")
    table.add_row("Sortino Ratio (Anualizado)", f"{m.sortino_ratio:.2f}")
    table.add_row("Calmar Ratio", f"{m.calmar_ratio:.2f}")
    table.add_row("CAGR", f"{m.cagr_pct * 100:.2f}%")
    table.add_row("Max Drawdown en Balance ($)", f"${m.max_drawdown_dollars:,.2f}")
    table.add_row("Max Drawdown en Balance (%)", f"{m.max_drawdown_pct * 100:.2f}%")
    table.add_row("Ratio Ganancia/Pérdida (Avg Win/Loss)", f"{m.win_loss_ratio:.2f}")

    console.print(table)

    target = prop_cfg["limits"]["profit_target"]
    max_dd_allowed = prop_cfg["limits"]["max_trailing_drawdown"]
    passed_target = m.net_pnl >= target
    survived_drawdown = m.max_drawdown_dollars < max_dd_allowed

    status_color = "green" if (passed_target and survived_drawdown) else "red"
    console.print("\n[bold]Evaluación de Cumplimiento de Firma de Fondeo:[/bold]")
    console.print(f" • Objetivo de Beneficio (${target:,.2f}): {'[green]ALCANZADO[/green]' if passed_target else '[yellow]NO ALCANZADO[/yellow]'}")
    console.print(f" • Límite Max Drawdown (${max_dd_allowed:,.2f}): {'[green]RESPETADO[/green]' if survived_drawdown else '[bold red]VIOLADO[/bold red]'}")

    if passed_target and survived_drawdown:
        console.print(f"[bold {status_color}]>>> RESULTADO: CUENTA APROBADA EXITOSAMENTE <<<[/bold {status_color}]\n")
    else:
        console.print(f"[bold {status_color}]>>> RESULTADO: REGLAS NO COMPLETADAS <<<[/bold {status_color}]\n")


if __name__ == "__main__":
    main()
