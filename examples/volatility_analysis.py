#!/usr/bin/env python3
"""
Volatility Analysis Deep Dive

This script focuses specifically on volatility analysis, comparing
different volatility measures and their characteristics.

Usage:
    python examples/volatility_analysis.py
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from stock_analyzer import StockAnalyzer
from plotting import StockPlotter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def main():
    """Run detailed volatility analysis."""
    
    print("🚀 Starting Volatility Analysis Deep Dive")
    print("=" * 50)
    
    # Initialize analyzer and plotter
    analyzer = StockAnalyzer()
    plotter = StockPlotter()
    
    # Configuration
    symbol = "AAPL"
    period = "3y"  # Longer period for volatility analysis
    
    print(f"📊 Analyzing: {symbol}")
    print(f"📅 Period: {period}")
    print("-" * 30)
    
    try:
        # Step 1: Fetch data
        print("1️⃣  Fetching stock data...")
        data = analyzer.fetch_stock_data(symbol, period=period)
        
        # Step 2: Calculate returns
        print("2️⃣  Calculating returns...")
        returns = analyzer.calculate_returns(data, method='both')
        
        # Step 3: Compare different volatility measures
        print("3️⃣  Calculating different volatility measures...")
        
        # EWMA with different decay factors
        ewma_094 = analyzer.calculate_ewma_volatility(returns, decay_factor=0.94, annualize=True)
        ewma_090 = analyzer.calculate_ewma_volatility(returns, decay_factor=0.90, annualize=True)
        ewma_097 = analyzer.calculate_ewma_volatility(returns, decay_factor=0.97, annualize=True)
        
        # Rolling volatilities with different windows
        rolling_20 = analyzer.calculate_rolling_volatility(returns, window=20, annualize=True)
        rolling_60 = analyzer.calculate_rolling_volatility(returns, window=60, annualize=True)
        rolling_252 = analyzer.calculate_rolling_volatility(returns, window=252, annualize=True)
        
        # Step 4: Create volatility comparison chart
        print("4️⃣  Creating volatility comparison visualization...")
        
        # Select simple returns for analysis
        simple_ret_col = [col for col in returns.columns if 'simple' in col][0]
        
        # Combine all volatility measures
        vol_comparison = pd.DataFrame({
            'EWMA_0.94': ewma_094[[col for col in ewma_094.columns if 'simple' in col][0]],
            'EWMA_0.90': ewma_090[[col for col in ewma_090.columns if 'simple' in col][0]],
            'EWMA_0.97': ewma_097[[col for col in ewma_097.columns if 'simple' in col][0]],
            'Rolling_20d': rolling_20[[col for col in rolling_20.columns if 'simple' in col][0]],
            'Rolling_60d': rolling_60[[col for col in rolling_60.columns if 'simple' in col][0]],
            'Rolling_252d': rolling_252[[col for col in rolling_252.columns if 'simple' in col][0]]
        })
        
        # Plot comparison
        plt.figure(figsize=(15, 10))
        
        # Volatility comparison
        plt.subplot(2, 2, 1)
        for col in vol_comparison.columns:
            plt.plot(vol_comparison.index, vol_comparison[col] * 100, 
                    label=col, linewidth=1.5)
        plt.title('Volatility Measures Comparison', fontweight='bold')
        plt.ylabel('Annualized Volatility (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Returns and volatility
        plt.subplot(2, 2, 2)
        returns_data = returns[simple_ret_col] * 100
        vol_data = ewma_094[[col for col in ewma_094.columns if 'simple' in col][0]] * 100
        
        plt.plot(returns_data.index, returns_data.values, alpha=0.6, color='blue', label='Daily Returns')
        plt.plot(vol_data.index, vol_data.values, color='red', linewidth=2, label='EWMA Volatility')
        plt.plot(vol_data.index, -vol_data.values, color='red', linewidth=2, linestyle='--')
        plt.title('Returns vs Volatility Envelope', fontweight='bold')
        plt.ylabel('Returns/Volatility (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Volatility distribution
        plt.subplot(2, 2, 3)
        plt.hist(vol_data.dropna() * 100, bins=50, alpha=0.7, density=True, color='orange')
        plt.title('Volatility Distribution', fontweight='bold')
        plt.xlabel('Annualized Volatility (%)')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        # Volatility persistence (autocorrelation)
        plt.subplot(2, 2, 4)
        vol_autocorr = [vol_data.autocorr(lag=i) for i in range(1, 21)]
        plt.bar(range(1, 21), vol_autocorr, alpha=0.7, color='green')
        plt.title('Volatility Autocorrelation', fontweight='bold')
        plt.xlabel('Lag (days)')
        plt.ylabel('Autocorrelation')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Step 5: Volatility statistics
        print("5️⃣  Volatility Statistics Summary:")
        print("=" * 50)
        
        vol_stats = vol_comparison.describe()
        vol_stats = vol_stats * 100  # Convert to percentage
        print("Volatility Measures Statistics (%):")
        print(vol_stats.round(2))
        
        # Volatility clustering analysis
        print(f"\n📊 Volatility Clustering Analysis:")
        squared_returns = returns[simple_ret_col] ** 2
        clustering_corr = squared_returns.autocorr(lag=1)
        print(f"Squared returns 1-day autocorrelation: {clustering_corr:.4f}")
        print("(Higher values indicate stronger volatility clustering)")
        
        # Extreme volatility events
        print(f"\n⚡ Extreme Volatility Events:")
        current_vol = vol_data.iloc[-1]
        high_vol_threshold = vol_data.quantile(0.95)
        low_vol_threshold = vol_data.quantile(0.05)
        
        print(f"Current volatility: {current_vol:.2f}%")
        print(f"95th percentile: {high_vol_threshold:.2f}%")
        print(f"5th percentile: {low_vol_threshold:.2f}%")
        
        high_vol_days = (vol_data > high_vol_threshold).sum()
        low_vol_days = (vol_data < low_vol_threshold).sum()
        
        print(f"High volatility days (>95th percentile): {high_vol_days}")
        print(f"Low volatility days (<5th percentile): {low_vol_days}")
        
        print("\n✅ Volatility analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return
    
    print("\n💡 Key Insights:")
    print("- EWMA volatility responds faster to market changes")
    print("- Different decay factors show different sensitivities")
    print("- Rolling volatility provides smoother estimates")
    print("- Volatility shows strong persistence (clustering)")
    print("- Extreme volatility events are relatively rare")


if __name__ == "__main__":
    main()