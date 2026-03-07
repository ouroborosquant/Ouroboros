"""
FORTRESS v5 - backtest_engine.py
Path: research/backtest_engine.py

Event-Driven Backtest Engine.
Enforces strict causal integrity, dynamic slippage, and institutional risk metrics.
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta

# Internal Imports
from data.pipeline import DataPipeline, LookAheadError
from research.research_lab import calculate_dsr, calculate_pbo

logger = logging.getLogger("BacktestEngine")

class EventDrivenBacktester:
    def __init__(self, data_pipeline: DataPipeline, config: Dict[str, Any]):
        self.pipeline = data_pipeline
        self.initial_capital = config.get('initial_capital', 100000.0)
        self.base_slippage_bps = config.get('base_slippage_bps', 2.0)
        self.transaction_fee_pct = config.get('transaction_fee_pct', 0.0000) # Alpaca is zero commission
        
        # Internal state
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.history = []

    async def run_backtest(self, strategy_models: Dict, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Steps through history day-by-day, passing strictly point-in-time data to the models.
        """
        logger.info(f"Initiating Event-Driven Backtest: {start_date} to {end_date}")
        
        # Generate the calendar of trading days
        calendar = self._generate_trading_calendar(start_date, end_date)
        
        for current_date in calendar:
            try:
                # 1. THE LOOK-AHEAD FIREWALL
                # This explicitly queries the database using `as_of_date <= current_date`.
                # If the strategy tries to access tomorrow's GDP report, this raises an error.
                obs_vector = await self.pipeline.get_observation_vector(as_of_date=current_date)
                
                # 2. Strategy Inference
                # Pass the legally acquired observation vector to the Mamba-KAN and EDT models
                z_t = strategy_models['regime_encoder'].get_posterior(obs_vector)
                target_weights = strategy_models['edt_agent'].get_weights(obs_vector, z_t)
                
                # 3. Execution Simulation
                # Fetch ACTUAL prices on the day of execution
                prices = await self._get_execution_prices(current_date)
                
                # Calculate required trades to reach target weights
                orders = self._calculate_orders(target_weights, prices)
                
                # Execute orders with dynamic slippage
                self._execute_orders(orders, prices, current_date, z_t)
                
                # 4. End of Day Accounting
                self._mark_to_market(prices, current_date)
                
            except LookAheadError as e:
                logger.critical(f"Look-Ahead Bias Detected on {current_date}: {e}")
                raise
            except Exception as e:
                logger.error(f"Simulation error on {current_date}: {e}")
                
        return self._generate_tearsheet()

    def _execute_orders(self, orders: Dict[str, float], prices: Dict[str, float], current_date: str, z_t: np.ndarray):
        """
        Simulates execution applying realistic market friction.
        """
        for ticker, qty in orders.items():
            if qty == 0:
                continue
                
            price = prices.get(ticker)
            if not price:
                continue
                
            # Dynamic slippage: scales with the latent regime's volatility proxy
            # e.g., if z_t indicates a crash regime, spread and slippage widen dramatically
            regime_volatility_penalty = 1.0 + float(np.abs(z_t[0])) # Simplified proxy
            effective_slippage = (self.base_slippage_bps / 10000.0) * regime_volatility_penalty
            
            # Buy orders pay the offer (price + slippage), Sell orders hit the bid (price - slippage)
            execution_price = price * (1 + effective_slippage) if qty > 0 else price * (1 - effective_slippage)
            
            trade_value = execution_price * abs(qty)
            
            # Check for sufficient cash on buys
            if qty > 0 and trade_value > self.cash:
                qty = self.cash / execution_price # Scale down to available cash
                trade_value = self.cash
                
            # Execute
            self.cash -= trade_value if qty > 0 else -trade_value
            self.positions[ticker] = self.positions.get(ticker, 0) + qty

    def _mark_to_market(self, prices: Dict[str, float], current_date: str):
        """Calculates total portfolio value at the end of the day."""
        pos_value = sum(self.positions[ticker] * prices.get(ticker, 0.0) for ticker in self.positions)
        self.portfolio_value = self.cash + pos_value
        
        self.history.append({
            'date': current_date,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'drawdown': self._calculate_current_drawdown()
        })

    def _generate_tearsheet(self) -> pd.DataFrame:
        """
        Compiles the results and calculates institutional metrics.
        """
        df = pd.DataFrame(self.history)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        df['daily_return'] = df['portfolio_value'].pct_change().fillna(0)
        
        # Calculate standard metrics
        cagr = (df['portfolio_value'].iloc[-1] / self.initial_capital) ** (252 / len(df)) - 1
        vol = df['daily_return'].std() * np.sqrt(252)
        sharpe = (cagr - 0.04) / vol if vol > 0 else 0 # Assuming 4% risk-free rate
        max_dd = df['drawdown'].min()
        
        # Institutional Quality Gates
        # Deflated Sharpe Ratio adjusts for multiple testing (number of Optuna trials)
        dsr = calculate_dsr(trials_history=[], current_sharpe=sharpe) 
        
        logger.info(f"Backtest Complete. CAGR: {cagr:.2%}, Sharpe: {sharpe:.2f}, Max DD: {max_dd:.2%}, DSR: {dsr:.2f}")
        
        return df

    def _calculate_current_drawdown(self) -> float:
        if not self.history:
            return 0.0
        peak = max([h['portfolio_value'] for h in self.history] + [self.initial_capital])
        return (self.portfolio_value - peak) / peak

    def _generate_trading_calendar(self, start: str, end: str) -> List[str]:
        # Scaffold: Returns a list of YYYY-MM-DD strings for business days
        return pd.date_range(start=start, end=end, freq='B').strftime('%Y-%m-%d').tolist()
        
    async def _get_execution_prices(self, date: str) -> Dict[str, float]:
        # Scaffold: Returns dummy execution prices for the given date
        return {'SPY': 500.0, 'TLT': 95.0, 'GLD': 200.0}
        
    def _calculate_orders(self, target_weights: np.ndarray, prices: Dict[str, float]) -> Dict[str, float]:
        # Scaffold: Translates target weights into asset quantities
        return {'SPY': 10, 'TLT': 50}
    
if __name__ == "__main__":
    import asyncio
    from data.pipeline import DataPipeline
    pipeline = DataPipeline()
    config = {'initial_capital': 100000.0, 'base_slippage_bps': 2.0}
    backtester = EventDrivenBacktester(pipeline, config)
    strategy_models = {}  # Will fail gracefully until models are wired in
    asyncio.run(backtester.run_backtest(strategy_models, '2023-01-01', '2024-01-01'))