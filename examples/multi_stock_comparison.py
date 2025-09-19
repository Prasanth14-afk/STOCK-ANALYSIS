#!/usr/bin/env python3
"""
Multi-Stock Comparison Analysis

This script compares multiple stocks side by side, analyzing their
risk-return characteristics and correlations.

Usage:
    python examples/multi_stock_comparison.py
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
    """Run multi-stock comparison analysis."""
    
    print("🚀 Starting Multi-Stock Comparison Analysis")
    print("=" * 60)
    
    # Initialize analyzer and plotter
    analyzer = StockAnalyzer()
    plotter = StockPlotter()
    
    # Configuration - Tech giants comparison
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    period = "2y"
    
    print(f"📊 Comparing: {', '.join(symbols)}")
    print(f"📅 Period: {period}")
    print("-" * 40)
    
    try:
        # Step 1: Fetch data for all symbols
        print("1️⃣  Fetching stock data for all symbols...")
        data = analyzer.fetch_stock_data(symbols, period=period)
        
        if data.empty:
            print("❌ No data received. Exiting.")
            return
        
        # Step 2: Calculate returns
        print("2️⃣  Calculating returns...")
        returns = analyzer.calculate_returns(data, method='simple')
        
        # Step 3: Calculate EWMA volatility
        print("3️⃣  Calculating EWMA volatility...")
        volatility = analyzer.calculate_ewma_volatility(returns)
        
        # Step 4: Calculate cumulative returns
        print("4️⃣  Calculating cumulative returns...")
        cum_returns = analyzer.calculate_cumulative_returns(returns)
        
        # Step 5: Create comparison visualizations
        print("5️⃣  Creating comparison visualizations...")
        
        # Complete analysis dashboard
        figures = plotter.plot_complete_analysis(data, returns, volatility, symbols)
        
        # Show all plots
        plt.show()
        
        # Step 6: Comparative analysis
        print("6️⃣  Comparative Analysis:")
        print("=" * 40)
        
        comparison_data = []
        
        for symbol in symbols:
            # Find the return column for this symbol
            ret_cols = [col for col in returns.columns if symbol in col and 'simple' in col]
            vol_cols = [col for col in volatility.columns if symbol in col and 'simple' in col]
            
            if ret_cols and vol_cols:
                ret_col = ret_cols[0]
                vol_col = vol_cols[0]
                
                # Calculate metrics
                returns_data = returns[ret_col]
                daily_return = returns_data.mean()
                annual_return = daily_return * 252
                annual_vol = volatility[vol_col].iloc[-1]
                sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
                total_return = (1 + returns_data).prod() - 1
                max_drawdown = returns_data.min()
                
                comparison_data.append({
                    'Symbol': symbol,
                    'Total Return (%)': total_return * 100,
                    'Annual Return (%)': annual_return * 100,
                    'Annual Volatility (%)': annual_vol * 100,
                    'Sharpe Ratio': sharpe_ratio,
                    'Max Daily Loss (%)': max_drawdown * 100
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(2)
        
        print("\n📊 PERFORMANCE COMPARISON TABLE:")
        print(comparison_df.to_string(index=False))
        
        # Best performers
        print(f"\n🏆 BEST PERFORMERS:")
        print(f"Highest Total Return: {comparison_df.loc[comparison_df['Total Return (%)'].idxmax(), 'Symbol']}")
        print(f"Highest Sharpe Ratio: {comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax(), 'Symbol']}")
        print(f"Lowest Volatility: {comparison_df.loc[comparison_df['Annual Volatility (%)'].idxmin(), 'Symbol']}")
        
        # Correlation analysis
        print(f"\n🔗 CORRELATION ANALYSIS:")
        simple_returns_df = returns[[col for col in returns.columns if 'simple' in col]].copy()
        simple_returns_df.columns = [col.split('_')[0] for col in simple_returns_df.columns]
        
        corr_matrix = simple_returns_df.corr()
        print("Correlation Matrix:")
        print(corr_matrix.round(3))
        
        # Average correlation
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        print(f"\nAverage correlation: {avg_corr:.3f}")
        
        print("\n✅ Multi-stock analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return
    
    print("\n💡 Insights:")
    print("- Tech stocks often show high positive correlations")
    print("- Higher returns typically come with higher volatility")
    print("- Sharpe ratios help identify risk-adjusted performance")
    print("- Diversification benefits may be limited within same sector")


def save_results_to_csv(comparison_df, filename="stock_comparison_results.csv"):
    """Save comparison results to CSV file."""
    filepath = os.path.join("data", filename)
    os.makedirs("data", exist_ok=True)
    comparison_df.to_csv(filepath, index=False)
    print(f"📁 Results saved to: {filepath}")


if __name__ == "__main__":
    main()