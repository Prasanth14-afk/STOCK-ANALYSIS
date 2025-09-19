"""
Plotting Utilities for Stock Analysis

This module provides comprehensive plotting functions for visualizing:
- Stock price data
- Returns analysis
- Volatility analysis
- Risk-return relationships
- Distribution analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class StockPlotter:
    """
    A comprehensive plotting class for stock market analysis visualizations.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the StockPlotter.
        
        Parameters:
        -----------
        figsize : tuple
            Default figure size for plots
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    def plot_price_data(self, 
                       data: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       title: str = "Stock Price Analysis") -> plt.Figure:
        """
        Plot stock price time series.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Stock price data
        columns : list, optional
            Specific columns to plot. If None, plots all 'Close' columns
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        if columns is None:
            columns = [col for col in data.columns if 'Close' in col or col == 'Close']
        
        fig, axes = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2))
        
        # Price plot
        for i, col in enumerate(columns):
            if col in data.columns:
                axes[0].plot(data.index, data[col], 
                           label=col.replace('_Close', ''), 
                           color=self.colors[i % len(self.colors)],
                           linewidth=1.5)
        
        axes[0].set_title(f"{title} - Price", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("Price ($)", fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volume plot (if available)
        volume_cols = [col for col in data.columns if 'Volume' in col]
        if volume_cols:
            for i, col in enumerate(volume_cols):
                axes[1].bar(data.index, data[col] / 1e6, 
                          label=col.replace('_Volume', ''), 
                          alpha=0.7,
                          color=self.colors[i % len(self.colors)])
            
            axes[1].set_title("Trading Volume", fontsize=14, fontweight='bold')
            axes[1].set_ylabel("Volume (Millions)", fontsize=12)
            axes[1].legend()
        else:
            # If no volume data, plot price again with different view
            for i, col in enumerate(columns):
                if col in data.columns:
                    pct_change = data[col].pct_change().rolling(30).mean() * 100
                    axes[1].plot(data.index, pct_change, 
                               label=f"{col.replace('_Close', '')} - 30d MA Return %", 
                               color=self.colors[i % len(self.colors)])
            
            axes[1].set_title("30-Day Moving Average Returns (%)", fontsize=14, fontweight='bold')
            axes[1].set_ylabel("Return (%)", fontsize=12)
            axes[1].legend()
        
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_returns_analysis(self, 
                             returns: pd.DataFrame,
                             title: str = "Returns Analysis") -> plt.Figure:
        """
        Plot comprehensive returns analysis.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Returns data
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        n_series = len(returns.columns)
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0] * 1.2, self.figsize[1] * 1.2))
        axes = axes.flatten()
        
        # 1. Time series of returns
        for i, col in enumerate(returns.columns):
            axes[0].plot(returns.index, returns[col] * 100, 
                        label=col.replace('_returns', ''), 
                        alpha=0.8,
                        color=self.colors[i % len(self.colors)])
        
        axes[0].set_title("Daily Returns Time Series", fontsize=12, fontweight='bold')
        axes[0].set_ylabel("Returns (%)", fontsize=10)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Returns distribution
        for i, col in enumerate(returns.columns):
            axes[1].hist(returns[col] * 100, bins=50, alpha=0.7, 
                        label=col.replace('_returns', ''),
                        color=self.colors[i % len(self.colors)])
        
        axes[1].set_title("Returns Distribution", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Returns (%)", fontsize=10)
        axes[1].set_ylabel("Frequency", fontsize=10)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Cumulative returns
        for i, col in enumerate(returns.columns):
            cum_returns = (1 + returns[col]).cumprod() - 1
            axes[2].plot(returns.index, cum_returns * 100, 
                        label=col.replace('_returns', ''),
                        linewidth=2,
                        color=self.colors[i % len(self.colors)])
        
        axes[2].set_title("Cumulative Returns", fontsize=12, fontweight='bold')
        axes[2].set_ylabel("Cumulative Returns (%)", fontsize=10)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. Rolling correlation (if multiple series)
        if n_series > 1:
            correlation_data = returns.rolling(window=60).corr().iloc[0::n_series, 1]
            axes[3].plot(correlation_data.index, correlation_data.values, 
                        linewidth=2, color='red')
            axes[3].set_title("60-Day Rolling Correlation", fontsize=12, fontweight='bold')
            axes[3].set_ylabel("Correlation", fontsize=10)
            axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        else:
            # Q-Q plot for single series
            from scipy import stats
            col = returns.columns[0]
            stats.probplot(returns[col], dist="norm", plot=axes[3])
            axes[3].set_title("Q-Q Plot (Normal Distribution)", fontsize=12, fontweight='bold')
        
        axes[3].grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    def plot_volatility_analysis(self, 
                                volatility: pd.DataFrame,
                                returns: Optional[pd.DataFrame] = None,
                                title: str = "Volatility Analysis") -> plt.Figure:
        """
        Plot volatility analysis.
        
        Parameters:
        -----------
        volatility : pd.DataFrame
            Volatility data
        returns : pd.DataFrame, optional
            Returns data for additional analysis
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0] * 1.2, self.figsize[1] * 1.2))
        axes = axes.flatten()
        
        # 1. Volatility time series
        for i, col in enumerate(volatility.columns):
            axes[0].plot(volatility.index, volatility[col] * 100, 
                        label=col.replace('_ewma_vol_annualized', '').replace('_returns', ''), 
                        linewidth=2,
                        color=self.colors[i % len(self.colors)])
        
        axes[0].set_title("EWMA Volatility Time Series", fontsize=12, fontweight='bold')
        axes[0].set_ylabel("Annualized Volatility (%)", fontsize=10)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Volatility distribution
        for i, col in enumerate(volatility.columns):
            axes[1].hist(volatility[col] * 100, bins=30, alpha=0.7, 
                        label=col.replace('_ewma_vol_annualized', '').replace('_returns', ''),
                        color=self.colors[i % len(self.colors)])
        
        axes[1].set_title("Volatility Distribution", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Annualized Volatility (%)", fontsize=10)
        axes[1].set_ylabel("Frequency", fontsize=10)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Volatility clustering (if returns available)
        if returns is not None and len(returns.columns) > 0:
            col = returns.columns[0]
            squared_returns = returns[col] ** 2
            axes[2].scatter(squared_returns.shift(1), squared_returns, 
                          alpha=0.6, s=10, color=self.colors[0])
            axes[2].set_title("Volatility Clustering", fontsize=12, fontweight='bold')
            axes[2].set_xlabel("Previous Day Squared Returns", fontsize=10)
            axes[2].set_ylabel("Current Day Squared Returns", fontsize=10)
        else:
            # Alternative: volatility autocorrelation
            if len(volatility.columns) > 0:
                vol_col = volatility.columns[0]
                vol_data = volatility[vol_col].dropna()
                autocorr = [vol_data.autocorr(lag=i) for i in range(1, 21)]
                axes[2].bar(range(1, 21), autocorr, color=self.colors[0], alpha=0.7)
                axes[2].set_title("Volatility Autocorrelation", fontsize=12, fontweight='bold')
                axes[2].set_xlabel("Lag (days)", fontsize=10)
                axes[2].set_ylabel("Autocorrelation", fontsize=10)
        
        axes[2].grid(True, alpha=0.3)
        
        # 4. Volatility statistics summary
        axes[3].axis('off')
        stats_text = "Volatility Statistics:\n\n"
        
        for col in volatility.columns:
            clean_name = col.replace('_ewma_vol_annualized', '').replace('_returns', '')
            vol_data = volatility[col] * 100
            stats_text += f"{clean_name}:\n"
            stats_text += f"  Mean: {vol_data.mean():.2f}%\n"
            stats_text += f"  Std: {vol_data.std():.2f}%\n"
            stats_text += f"  Min: {vol_data.min():.2f}%\n"
            stats_text += f"  Max: {vol_data.max():.2f}%\n"
            stats_text += f"  Current: {vol_data.iloc[-1]:.2f}%\n\n"
        
        axes[3].text(0.1, 0.9, stats_text, transform=axes[3].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_risk_return_scatter(self, 
                               returns: pd.DataFrame,
                               volatility: pd.DataFrame,
                               title: str = "Risk-Return Analysis") -> plt.Figure:
        """
        Plot risk-return scatter plot.
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Returns data
        volatility : pd.DataFrame
            Volatility data
        title : str
            Plot title
            
        Returns:
        --------
        plt.Figure
            Matplotlib figure object
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Calculate annualized returns and volatilities
        risk_return_data = []
        
        for ret_col in returns.columns:
            # Find corresponding volatility column
            vol_col = None
            for v_col in volatility.columns:
                if ret_col.replace('_returns', '') in v_col:
                    vol_col = v_col
                    break
            
            if vol_col is not None:
                ann_return = returns[ret_col].mean() * 252 * 100
                ann_vol = volatility[vol_col].iloc[-1] * 100
                sharpe = ann_return / ann_vol if ann_vol != 0 else 0
                
                risk_return_data.append({
                    'symbol': ret_col.replace('_simple_returns', '').replace('_log_returns', ''),
                    'return': ann_return,
                    'volatility': ann_vol,
                    'sharpe': sharpe
                })
        
        if risk_return_data:
            # Create scatter plot
            for i, data in enumerate(risk_return_data):
                ax.scatter(data['volatility'], data['return'], 
                          s=100, alpha=0.7, 
                          color=self.colors[i % len(self.colors)],
                          label=f"{data['symbol']} (Sharpe: {data['sharpe']:.2f})")
                
                # Add text annotation
                ax.annotate(data['symbol'], 
                           (data['volatility'], data['return']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8)
            
            ax.set_xlabel("Annualized Volatility (%)", fontsize=12)
            ax.set_ylabel("Annualized Return (%)", fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add quadrant lines
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.5, 
                      color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_complete_analysis(self, 
                             data: pd.DataFrame,
                             returns: pd.DataFrame,
                             volatility: pd.DataFrame,
                             symbols: List[str]) -> List[plt.Figure]:
        """
        Create a complete analysis dashboard with all plots.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data
        returns : pd.DataFrame
            Returns data
        volatility : pd.DataFrame
            Volatility data
        symbols : list
            List of stock symbols
            
        Returns:
        --------
        List[plt.Figure]
            List of all created figures
        """
        figures = []
        
        # 1. Price analysis
        fig1 = self.plot_price_data(data, title=f"Stock Analysis - {', '.join(symbols)}")
        figures.append(fig1)
        
        # 2. Returns analysis
        fig2 = self.plot_returns_analysis(returns, title=f"Returns Analysis - {', '.join(symbols)}")
        figures.append(fig2)
        
        # 3. Volatility analysis
        fig3 = self.plot_volatility_analysis(volatility, returns, 
                                           title=f"Volatility Analysis - {', '.join(symbols)}")
        figures.append(fig3)
        
        # 4. Risk-return analysis
        fig4 = self.plot_risk_return_scatter(returns, volatility, 
                                           title=f"Risk-Return Analysis - {', '.join(symbols)}")
        figures.append(fig4)
        
        return figures
    
    def save_all_plots(self, figures: List[plt.Figure], 
                      output_dir: str = "./plots", 
                      prefix: str = "stock_analysis") -> None:
        """
        Save all plots to files.
        
        Parameters:
        -----------
        figures : List[plt.Figure]
            List of figures to save
        output_dir : str
            Output directory
        prefix : str
            Filename prefix
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        plot_names = ["price_analysis", "returns_analysis", "volatility_analysis", "risk_return"]
        
        for i, fig in enumerate(figures):
            if i < len(plot_names):
                filename = f"{prefix}_{plot_names[i]}.png"
                filepath = os.path.join(output_dir, filename)
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                print(f"Saved: {filepath}")
        
        print(f"All plots saved to {output_dir}")


def quick_plot(data: pd.DataFrame, 
              returns: Optional[pd.DataFrame] = None,
              volatility: Optional[pd.DataFrame] = None) -> None:
    """
    Quick plotting function for immediate visualization.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Stock price data
    returns : pd.DataFrame, optional
        Returns data
    volatility : pd.DataFrame, optional
        Volatility data
    """
    plotter = StockPlotter()
    
    # Plot price data
    plotter.plot_price_data(data)
    plt.show()
    
    # Plot returns if available
    if returns is not None:
        plotter.plot_returns_analysis(returns)
        plt.show()
    
    # Plot volatility if available
    if volatility is not None:
        plotter.plot_volatility_analysis(volatility, returns)
        plt.show()