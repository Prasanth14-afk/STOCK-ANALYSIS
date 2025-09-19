"""
Stock Analyzer Module

This module provides comprehensive functionality for stock market analysis including:
- Fetching stock data from Yahoo Finance
- Calculating various types of returns
- Computing EWMA volatility
- Statistical analysis of time series
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Dict, Tuple, Optional
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class StockAnalyzer:
    """
    A comprehensive stock analysis class for fetching data, calculating returns,
    and analyzing volatility using Yahoo Finance data.
    """
    
    def __init__(self):
        """Initialize the StockAnalyzer."""
        self.data = None
        self.returns = None
        self.volatility = None
        
    def fetch_stock_data(self, 
                        symbols: Union[str, list], 
                        period: str = "2y",
                        start: Optional[str] = None,
                        end: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        
        Parameters:
        -----------
        symbols : str or list
            Stock symbol(s) to fetch (e.g., 'AAPL' or ['AAPL', 'GOOGL'])
        period : str
            Period to fetch data for ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        start : str, optional
            Start date in 'YYYY-MM-DD' format
        end : str, optional
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        pd.DataFrame
            Stock data with OHLCV columns
        """
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
            
            data_frames = {}
            
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                
                if start and end:
                    hist = ticker.history(start=start, end=end)
                else:
                    hist = ticker.history(period=period)
                
                if hist.empty:
                    print(f"Warning: No data found for symbol {symbol}")
                    continue
                    
                # Clean the data
                hist = hist.dropna()
                
                # If multiple symbols, add symbol to column names
                if len(symbols) > 1:
                    hist.columns = [f"{symbol}_{col}" for col in hist.columns]
                
                data_frames[symbol] = hist
            
            if len(data_frames) == 1:
                self.data = list(data_frames.values())[0]
            else:
                self.data = pd.concat(data_frames.values(), axis=1)
            
            print(f"Successfully fetched data for {len(data_frames)} symbol(s)")
            print(f"Date range: {self.data.index[0].strftime('%Y-%m-%d')} to {self.data.index[-1].strftime('%Y-%m-%d')}")
            print(f"Total trading days: {len(self.data)}")
            
            return self.data
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_returns(self, 
                         data: Optional[pd.DataFrame] = None,
                         method: str = 'simple',
                         price_column: str = 'Close') -> pd.DataFrame:
        """
        Calculate stock returns.
        
        Parameters:
        -----------
        data : pd.DataFrame, optional
            Stock data. If None, uses self.data
        method : str
            Return calculation method: 'simple', 'log', or 'both'
        price_column : str
            Column name for price data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with calculated returns
        """
        if data is None:
            data = self.data
            
        if data is None or data.empty:
            raise ValueError("No data available. Please fetch data first.")
        
        returns_data = {}
        
        # Handle multiple symbols
        price_columns = [col for col in data.columns if price_column in col or col == price_column]
        
        if not price_columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
        
        for col in price_columns:
            prices = data[col].dropna()
            
            if method in ['simple', 'both']:
                simple_returns = prices.pct_change().dropna()
                returns_data[f"{col}_simple_returns"] = simple_returns
            
            if method in ['log', 'both']:
                log_returns = np.log(prices / prices.shift(1)).dropna()
                returns_data[f"{col}_log_returns"] = log_returns
        
        self.returns = pd.DataFrame(returns_data)
        
        # Calculate summary statistics
        print("Return Statistics:")
        print("-" * 50)
        for col in self.returns.columns:
            stats = {
                'Mean (daily)': self.returns[col].mean(),
                'Std (daily)': self.returns[col].std(),
                'Annualized Mean': self.returns[col].mean() * 252,
                'Annualized Std': self.returns[col].std() * np.sqrt(252),
                'Sharpe Ratio': (self.returns[col].mean() * 252) / (self.returns[col].std() * np.sqrt(252)),
                'Skewness': self.returns[col].skew(),
                'Kurtosis': self.returns[col].kurtosis()
            }
            
            print(f"\n{col}:")
            for stat_name, value in stats.items():
                print(f"  {stat_name}: {value:.4f}")
        
        return self.returns
    
    def calculate_ewma_volatility(self, 
                                 returns: Optional[pd.DataFrame] = None,
                                 decay_factor: float = 0.94,
                                 annualize: bool = True) -> pd.DataFrame:
        """
        Calculate Exponentially Weighted Moving Average (EWMA) volatility.
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
            Returns data. If None, uses self.returns
        decay_factor : float
            Decay factor for EWMA (typically 0.94 for daily data)
        annualize : bool
            Whether to annualize the volatility
            
        Returns:
        --------
        pd.DataFrame
            EWMA volatility
        """
        if returns is None:
            returns = self.returns
            
        if returns is None or returns.empty:
            raise ValueError("No returns data available. Please calculate returns first.")
        
        volatility_data = {}
        
        for col in returns.columns:
            # Calculate EWMA variance
            ewma_var = returns[col].ewm(alpha=1-decay_factor).var()
            
            # Calculate volatility (standard deviation)
            ewma_vol = np.sqrt(ewma_var)
            
            if annualize:
                ewma_vol = ewma_vol * np.sqrt(252)
                col_name = f"{col}_ewma_vol_annualized"
            else:
                col_name = f"{col}_ewma_vol_daily"
            
            volatility_data[col_name] = ewma_vol
        
        self.volatility = pd.DataFrame(volatility_data)
        
        print("EWMA Volatility Statistics:")
        print("-" * 50)
        for col in self.volatility.columns:
            print(f"{col}:")
            print(f"  Mean: {self.volatility[col].mean():.4f}")
            print(f"  Min: {self.volatility[col].min():.4f}")
            print(f"  Max: {self.volatility[col].max():.4f}")
            print(f"  Current: {self.volatility[col].iloc[-1]:.4f}")
        
        return self.volatility
    
    def calculate_rolling_volatility(self, 
                                   returns: Optional[pd.DataFrame] = None,
                                   window: int = 30,
                                   annualize: bool = True) -> pd.DataFrame:
        """
        Calculate rolling window volatility.
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
            Returns data. If None, uses self.returns
        window : int
            Rolling window size in days
        annualize : bool
            Whether to annualize the volatility
            
        Returns:
        --------
        pd.DataFrame
            Rolling volatility
        """
        if returns is None:
            returns = self.returns
            
        if returns is None or returns.empty:
            raise ValueError("No returns data available. Please calculate returns first.")
        
        rolling_vol_data = {}
        
        for col in returns.columns:
            rolling_vol = returns[col].rolling(window=window).std()
            
            if annualize:
                rolling_vol = rolling_vol * np.sqrt(252)
                col_name = f"{col}_rolling_vol_{window}d_annualized"
            else:
                col_name = f"{col}_rolling_vol_{window}d_daily"
            
            rolling_vol_data[col_name] = rolling_vol
        
        return pd.DataFrame(rolling_vol_data)
    
    def calculate_cumulative_returns(self, 
                                   returns: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate cumulative returns.
        
        Parameters:
        -----------
        returns : pd.DataFrame, optional
            Returns data. If None, uses self.returns
            
        Returns:
        --------
        pd.DataFrame
            Cumulative returns
        """
        if returns is None:
            returns = self.returns
            
        if returns is None or returns.empty:
            raise ValueError("No returns data available. Please calculate returns first.")
        
        cumulative_returns = {}
        
        for col in returns.columns:
            if 'simple' in col.lower():
                # For simple returns: (1 + r1) * (1 + r2) * ... - 1
                cum_ret = (1 + returns[col]).cumprod() - 1
            else:
                # For log returns: cumulative sum
                cum_ret = returns[col].cumsum()
            
            cumulative_returns[f"{col}_cumulative"] = cum_ret
        
        return pd.DataFrame(cumulative_returns)
    
    def get_summary_statistics(self) -> Dict:
        """
        Get comprehensive summary statistics for the analysis.
        
        Returns:
        --------
        Dict
            Dictionary containing various statistics
        """
        if self.data is None:
            return {"error": "No data available"}
        
        summary = {
            "data_info": {
                "start_date": self.data.index[0].strftime('%Y-%m-%d'),
                "end_date": self.data.index[-1].strftime('%Y-%m-%d'),
                "total_days": len(self.data),
                "columns": list(self.data.columns)
            }
        }
        
        if self.returns is not None:
            summary["returns_info"] = {
                "total_return_series": len(self.returns.columns),
                "return_columns": list(self.returns.columns)
            }
        
        if self.volatility is not None:
            summary["volatility_info"] = {
                "volatility_series": len(self.volatility.columns),
                "volatility_columns": list(self.volatility.columns)
            }
        
        return summary