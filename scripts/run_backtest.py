import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
import yaml

from src.alpha.trend_continuation import ORBTrendContinuationStrategy
from src.backtest.engine import BacktestEngine
from src.backtest.fee_models import CMEFeeModel
from src.backtest.pass_probability import arrays_from_trades, pass_probability
from src.core.events import BarEvent
from src.risk.position_sizer import InstrumentSpecs


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecutor de Backtests CME ORB - Datos Reales")
    parser.add_argument("--data", type=str, default="data/raw/minute/cme_mnq_1m.parquet")
    parser.add_argument("--instrument", type=str, default="MNQ")
    parser.add_argument("--prop-config", type=str, default="configs/prop_firm/evaluation_50k.yaml")
    parser.add_argument("--strategy-config", type=str, default="configs/strategy/orb_base.yaml")
    args = parser.parse_args()

    console = Console()

    with open(args.prop_config) as f:
        prop_cfg = yaml.safe_load(f)
    with open(args.strategy_config) as f:
        strat_cfg = yaml.safe_load(f)

    specs = InstrumentSpecs(
        symbol=args.instrument,
        tick_size=0.25,
        point_value=2.00,
        max_contracts=prop_cfg["limits"]["max_open_contracts"].get(args.instrument, 6),
    )

    fee_model = CMEFeeModel(commission_per_contract=0.62, slippage_ticks=1.0)

    console.print("[bold cyan][*] Inicializando ORBTrendContinuationStrategy para datos de mercado reales...[/bold cyan]")
    strategy = ORBTrendContinuationStrategy(
        symbol=args.instrument,
        buffer_ticks=strat_cfg["opening_range"]["buffer_ticks"],
        retest_tolerance_ticks=12,
        min_rvol_threshold=strat_cfg["entry_filters"].get("min_rvol_threshold", 0.8),
        stop_atr_multiplier=strat_cfg["exit_rules"]["stop_loss"]["stop_atr_multiplier"],
        risk_reward_ratio=strat_cfg["exit_rules"]["take_profit"]["risk_reward_ratio"],
        hard_stop_max_ticks=strat_cfg["exit_rules"]["stop_loss"]["hard_stop_max_ticks"],
        stop_min_ticks=strat_cfg["exit_rules"]["stop_loss"].get("min_stop_ticks", 40),
        max_wait_bars=strat_cfg["entry_filters"].get("max_retest_bars", 25),
        allow_vwap_retest=strat_cfg["entry_filters"].get("allow_vwap_retest", False),
        rvol_min_history=strat_cfg["entry_filters"].get("rvol_min_history", 2),
    )

    engine = BacktestEngine(
        strategy=strategy,
        specs=specs,
        fee_model=fee_model,
        initial_capital=prop_cfg["initial_balance"],
        max_trailing_drawdown=prop_cfg["limits"]["max_trailing_drawdown"],
        max_daily_loss=prop_cfg["limits"]["max_daily_loss_limit"],
        debug=False,
        max_cost_ratio=strat_cfg["exit_rules"]["stop_loss"].get("max_cost_ratio", 0.12),
        drawdown_mode="peak_unrealized",
        lock_offset=100.0,
        stop_on_breach=True,
    )

    console.print(f"[bold blue][*] Leyendo datos históricos reales desde {args.data}...[/bold blue]")
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

    console.print(f"[bold blue][*] Ejecutando simulación intra-barra MTM sobre {len(bars):,} barras reales...[/bold blue]")
    result = engine.run(bars)
    m = result.metrics

    r_multiples = [getattr(t, "r_multiple", 0.0) for t in result.trades if hasattr(t, "r_multiple")]
    avg_r = np.mean(r_multiples) if r_multiples else 0.0

    table = Table(title="📊 Resultados de Mercado Real (CME Globex NQ/MNQ)")
    table.add_column("Métrica Cuantitativa", style="cyan", no_wrap=True)
    table.add_column("Valor Obtenido", style="bold green")

    table.add_row("Total de Operaciones", str(m.total_trades))
    table.add_row("Win Rate (incluye scale-outs/BE)", f"{m.win_rate * 100:.2f}%")
    table.add_row("Expectancy en R", f"{avg_r:+.3f} R" if r_multiples else "N/A")
    table.add_row("Expectancy por Trade ($)", f"${m.expectancy_dollars:,.2f}")
    table.add_row("Profit Factor", f"{m.profit_factor:.2f}")
    table.add_row("Beneficio Neto (Net PnL)", f"${m.net_pnl:,.2f}")
    table.add_row("Fricciones Totales", f"${m.total_commissions + m.total_slippage:,.2f}")
    table.add_row("Max Drawdown Intradía MTM ($)", f"${m.max_drawdown_dollars:,.2f}")
    table.add_row("Max Drawdown Intradía MTM (%)", f"{m.max_drawdown_pct * 100:.2f}%")
    table.add_row("Sharpe Ratio (Anualizado)", f"{m.sharpe_ratio:.2f}")
    table.add_row("Sortino Ratio (Anualizado)", f"{m.sortino_ratio:.2f}")

    console.print(table)

    if result.trades:
        contract_counts = Counter(t.contracts for t in result.trades)
        console.print("\n[bold yellow]Distribución de contratos por trade:[/bold yellow]")
        for k in sorted(contract_counts.keys()):
            console.print(f" • {k} contrato(s): {contract_counts[k]} trades ({contract_counts[k]/len(result.trades)*100:.1f}%)")

    target = prop_cfg["limits"]["profit_target"]
    max_dd_allowed = prop_cfg["limits"]["max_trailing_drawdown"]
    passed_target = m.net_pnl >= target
    breach = (
        getattr(result, "breach_occurred", False)
        or getattr(result, "breached", False)
        or (m.max_drawdown_dollars >= max_dd_allowed)
    )

    console.print("\n[bold]Auditoría de Evaluación Prop Firm:[/bold]")
    console.print(f" • Objetivo de Beneficio (${target:,.2f}): {'[green]ALCANZADO[/green]' if passed_target else '[yellow]NO ALCANZADO[/yellow]'}")
    console.print(f" • Trailing Floor MTM Breach: {'[bold red]BREACH DETECTADO (CUENTA FALLADA)[/bold red]' if breach else '[bold green]RESPETADO[/bold green]'}")


if __name__ == "__main__":
    main()
