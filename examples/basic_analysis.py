#!/usr/bin/env python3
"""
Basic Stock Analysis Example

This script demonstrates a simple stock analysis workflow using our
custom StockAnalyzer and StockPlotter classes.

Usage:
    python examples/basic_analysis.py
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from stock_analyzer import StockAnalyzer
from plotting import StockPlotter
import matplotlib.pyplot as plt


def main():
    """Run basic stock analysis example."""
    
    print("🚀 Starting Basic Stock Analysis Example")
    print("=" * 50)
    
    # Initialize analyzer and plotter
    analyzer = StockAnalyzer()
    plotter = StockPlotter()
    
    # Configuration
    symbol = "AAPL"  # Apple Inc.
    period = "1y"    # 1 year of data
    
    print(f"📊 Analyzing: {symbol}")
    print(f"📅 Period: {period}")
    print("-" * 30)
    
    try:
        # Step 1: Fetch data
        print("1️⃣  Fetching stock data...")
        data = analyzer.fetch_stock_data(symbol, period=period)
        
        if data.empty:
            print("❌ No data received. Exiting.")
            return
        
        # Step 2: Calculate returns
        print("2️⃣  Calculating returns...")
        returns = analyzer.calculate_returns(data, method='simple')
        
        # Step 3: Calculate EWMA volatility
        print("3️⃣  Calculating EWMA volatility...")
        volatility = analyzer.calculate_ewma_volatility(returns)
        
        # Step 4: Generate summary statistics
        print("4️⃣  Generating summary statistics...")
        summary = analyzer.get_summary_statistics()
        
        # Step 5: Create visualizations
        print("5️⃣  Creating visualizations...")
        
        # Price chart
        price_fig = plotter.plot_price_data(data, title=f"{symbol} Stock Analysis")
        
        # Returns analysis
        returns_fig = plotter.plot_returns_analysis(returns, title=f"{symbol} Returns Analysis")
        
        # Volatility analysis
        vol_fig = plotter.plot_volatility_analysis(volatility, returns, 
                                                  title=f"{symbol} Volatility Analysis")
        
        # Show all plots
        plt.show()
        
        # Step 6: Print key metrics
        print("6️⃣  Key Metrics Summary:")
        print("=" * 30)
        
        ret_col = returns.columns[0]
        vol_col = volatility.columns[0]
        
        # Calculate key metrics
        daily_return = returns[ret_col].mean()
        annual_return = daily_return * 252
        annual_vol = volatility[vol_col].iloc[-1]
        sharpe_ratio = annual_return / annual_vol if annual_vol != 0 else 0
        total_return = (1 + returns[ret_col]).prod() - 1
        
        print(f"Symbol: {symbol}")
        print(f"Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"Total Return: {total_return*100:.2f}%")
        print(f"Annualized Return: {annual_return*100:.2f}%")
        print(f"Annualized Volatility: {annual_vol*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"Current Price: ${data['Close'].iloc[-1]:.2f}")
        print(f"Max Daily Gain: {returns[ret_col].max()*100:.2f}%")
        print(f"Max Daily Loss: {returns[ret_col].min()*100:.2f}%")
        
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return
    
    print("\n💡 Tips:")
    print("- Try different symbols by changing the 'symbol' variable")
    print("- Adjust the period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')")
    print("- Check out the Jupyter notebook for more detailed analysis")


if __name__ == "__main__":
    main()