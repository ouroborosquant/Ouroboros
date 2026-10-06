import math
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    total_trades: int
    win_rate: float
    profit_factor: float
    expectancy_dollars: float
    gross_pnl: float
    net_pnl: float
    total_commissions: float
    total_slippage: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_dollars: float
    max_drawdown_pct: float
    avg_win: float
    avg_loss: float
    win_loss_ratio: float


class MetricsCalculator:
    @staticmethod
    def calculate(
        trades: list[dict],
        daily_equity_series: pd.Series,
        initial_capital: float = 50000.0,
        risk_free_rate: float = 0.04,
    ) -> PerformanceMetrics:
        if not trades or daily_equity_series.empty:
            return PerformanceMetrics(
                0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            )

        df_trades = pd.DataFrame(trades)
        net_pnls = df_trades["net_pnl"].values

        total_trades = len(net_pnls)
        wins = net_pnls[net_pnls > 0]
        losses = net_pnls[net_pnls < 0]

        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = abs(float(np.mean(losses))) if len(losses) > 0 else 0.0
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
        expectancy = float(np.mean(net_pnls))

        daily_returns = daily_equity_series.pct_change().dropna().values
        trading_days = len(daily_returns)

        rf_daily = risk_free_rate / 252.0
        excess_returns = daily_returns - rf_daily
        mean_excess = np.mean(excess_returns) if len(excess_returns) > 0 else 0.0
        std_returns = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 0.0
        sharpe = (mean_excess / std_returns * math.sqrt(252)) if std_returns > 0 else 0.0

        downside_returns = daily_returns[daily_returns < 0]
        downside_std = (
            math.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0.0
        )
        sortino = (mean_excess / downside_std * math.sqrt(252)) if downside_std > 0 else 0.0

        running_max = daily_equity_series.cummax()
        drawdowns = running_max - daily_equity_series
        max_dd_dollars = float(drawdowns.max())
        max_dd_pct = float((drawdowns / running_max).max())

        years = max(trading_days / 252.0, 0.05)
        final_equity = daily_equity_series.iloc[-1]
        cagr = ((final_equity / initial_capital) ** (1.0 / years) - 1.0) if final_equity > 0 else -1.0
        calmar = (cagr / max_dd_pct) if max_dd_pct > 0 else 0.0

        return PerformanceMetrics(
            total_trades=total_trades,
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 2),
            expectancy_dollars=round(expectancy, 2),
            gross_pnl=round(float(df_trades["gross_pnl"].sum()), 2),
            net_pnl=round(float(df_trades["net_pnl"].sum()), 2),
            total_commissions=round(float(df_trades["commission"].sum()), 2),
            total_slippage=round(float(df_trades["slippage"].sum()), 2),
            cagr_pct=round(cagr, 4),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            max_drawdown_dollars=round(max_dd_dollars, 2),
            max_drawdown_pct=round(max_dd_pct, 4),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            win_loss_ratio=round(win_loss_ratio, 2),
        )
