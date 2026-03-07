"""
FORTRESS v5 - visualize_tearsheet.py
Path: scripts/visualize_tearsheet.py

Reads the backtest results CSV and generates an institutional-grade 
performance tear sheet (Cumulative Returns, Drawdowns, and Rolling Volatility).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TearsheetGen")

def generate_tearsheet(csv_path: str = "backtest_results.csv", output_path: str = "tearsheet.png"):
    if not os.path.exists(csv_path):
        logger.error(f"Cannot find {csv_path}. Did the backtest complete successfully?")
        return

    logger.info("Generating Institutional Tear Sheet...")
    
    # Load data
    df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
    
    # Calculate derived metrics
    df['cum_return'] = df['portfolio_value'] / df['portfolio_value'].iloc[0]
    df['rolling_vol'] = df['daily_return'].rolling(window=21).std() * np.sqrt(252)
    
    # Set up the matplotlib figure layout
    plt.style.use('dark_background') # Institutional dark mode
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
    
    # --- Plot 1: Cumulative Returns ---
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df['cum_return'], color='#00ffcc', linewidth=2, label='FORTRESS v5')
    ax1.set_title('Cumulative Portfolio Return', fontsize=14, fontweight='bold', color='white')
    ax1.set_ylabel('Growth of $1', fontsize=12)
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc='upper left')
    
    # --- Plot 2: Underwater Drawdown ---
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(df.index, df['drawdown'], 0, color='#ff3366', alpha=0.5)
    ax2.plot(df.index, df['drawdown'], color='#ff3366', linewidth=1)
    ax2.set_title('Underwater Drawdown', fontsize=14, fontweight='bold', color='white')
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.2)
    
    # --- Plot 3: Rolling Volatility ---
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, df['rolling_vol'], color='#ffcc00', linewidth=1.5)
    ax3.set_title('21-Day Rolling Annualized Volatility', fontsize=14, fontweight='bold', color='white')
    ax3.set_ylabel('Volatility', fontsize=12)
    ax3.grid(True, alpha=0.2)
    
    # Final formatting and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Tear sheet successfully saved to {output_path}")

if __name__ == "__main__":
    generate_tearsheet()