"""
Data visualization module for financial text-to-SQL framework
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinancialDataVisualizer:
    def __init__(self, db_path: str = "financial_data.db"):
        """
        Initialize the visualizer with database connection
        """
        self.db_path = db_path
        self.colors = {
            'AAPL': '#007AFF',
            'TSLA': '#FF3B30', 
            'GOOGL': '#34C759',
            'MSFT': '#5856D6',
            'AMZN': '#FF9500',
            'META': '#AF52DE',
            'NFLX': '#FF2D92',
            'SPY': '#1C1C1E',
            'NVDA': '#00C7BE',
            'default': '#007AFF'
        }
    
    def get_stock_data(self, ticker: str, days: int = 365) -> pd.DataFrame:
        """
        Get stock data from database
        """
        conn = sqlite3.connect(self.db_path)
        query = f"""
        SELECT Date, Open, High, Low, Close, Volume, move 
        FROM stock_data 
        WHERE ticker = '{ticker}' 
        AND Date >= date('now', '-{days} days')
        ORDER BY Date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        return df
    
    def plot_price_history(self, tickers: List[str], days: int = 365, 
                          plot_type: str = 'close', figsize: Tuple[int, int] = (12, 8)):
        """
        Plot price history for multiple stocks
        
        Args:
            tickers: List of stock symbols
            days: Number of days to plot
            plot_type: 'close', 'ohlc', or 'candlestick'
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for ticker in tickers:
            df = self.get_stock_data(ticker, days)
            if not df.empty:
                color = self.colors.get(ticker, self.colors['default'])
                
                if plot_type == 'close':
                    ax.plot(df.index, df['Close'], label=ticker, color=color, linewidth=2)
                elif plot_type == 'ohlc':
                    ax.plot(df.index, df['High'], label=f'{ticker} High', color=color, alpha=0.7)
                    ax.plot(df.index, df['Low'], label=f'{ticker} Low', color=color, alpha=0.7)
                    ax.fill_between(df.index, df['Low'], df['High'], alpha=0.2, color=color)
        
        ax.set_title(f'Stock Price History - {plot_type.upper()} Prices', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_returns_distribution(self, tickers: List[str], days: int = 365, 
                                 figsize: Tuple[int, int] = (12, 8)):
        """
        Plot returns distribution for multiple stocks
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, ticker in enumerate(tickers[:4]):  # Limit to 4 stocks
            df = self.get_stock_data(ticker, days)
            if not df.empty:
                ax = axes[i]
                
                # Plot histogram
                ax.hist(df['move'], bins=30, alpha=0.7, color=self.colors.get(ticker, self.colors['default']))
                ax.axvline(df['move'].mean(), color='red', linestyle='--', label=f'Mean: {df["move"].mean():.2f}%')
                ax.axvline(df['move'].median(), color='green', linestyle='--', label=f'Median: {df["move"].median():.2f}%')
                
                ax.set_title(f'{ticker} Daily Returns Distribution', fontweight='bold')
                ax.set_xlabel('Daily Return (%)')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_volatility_comparison(self, tickers: List[str], days: int = 365, 
                                  window: int = 30, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot rolling volatility comparison
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for ticker in tickers:
            df = self.get_stock_data(ticker, days)
            if not df.empty:
                # Calculate rolling volatility
                volatility = df['move'].rolling(window=window).std() * np.sqrt(252)  # Annualized
                color = self.colors.get(ticker, self.colors['default'])
                ax.plot(df.index, volatility, label=ticker, color=color, linewidth=2)
        
        ax.set_title(f'Rolling Volatility Comparison ({window}-day window)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Annualized Volatility', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, tickers: List[str], days: int = 365, 
                                figsize: Tuple[int, int] = (10, 8)):
        """
        Plot correlation heatmap for multiple stocks
        """
        # Get data for all tickers
        all_data = {}
        for ticker in tickers:
            df = self.get_stock_data(ticker, days)
            if not df.empty:
                all_data[ticker] = df['move']
        
        if len(all_data) < 2:
            print("Need at least 2 stocks for correlation analysis")
            return
        
        # Create correlation matrix
        corr_df = pd.DataFrame(all_data)
        corr_matrix = corr_df.corr()
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax, fmt='.2f')
        
        ax.set_title(f'Stock Returns Correlation Matrix ({days} days)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_volume_analysis(self, ticker: str, days: int = 365, 
                           figsize: Tuple[int, int] = (12, 8)):
        """
        Plot volume analysis for a single stock
        """
        df = self.get_stock_data(ticker, days)
        if df.empty:
            print(f"No data found for {ticker}")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Price and Volume
        color = self.colors.get(ticker, self.colors['default'])
        ax1.plot(df.index, df['Close'], color=color, linewidth=2, label='Close Price')
        ax1.set_ylabel('Price ($)', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax1_twin = ax1.twinx()
        ax1_twin.bar(df.index, df['Volume'], alpha=0.3, color='gray', label='Volume')
        ax1_twin.set_ylabel('Volume', color='gray')
        ax1_twin.tick_params(axis='y', labelcolor='gray')
        
        ax1.set_title(f'{ticker} Price and Volume Analysis', fontweight='bold')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Plot 2: Volume distribution
        ax2.hist(df['Volume'], bins=30, alpha=0.7, color=color)
        ax2.axvline(df['Volume'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["Volume"].mean():,.0f}')
        ax2.set_xlabel('Volume')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_sql_query_results(self, sql_query: str, plot_type: str = 'auto', 
                              figsize: Tuple[int, int] = (12, 8)):
        """
        Execute SQL query and plot results
        
        Args:
            sql_query: SQL query to execute
            plot_type: 'auto', 'line', 'bar', 'scatter', 'histogram'
            figsize: Figure size
        """
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            if df.empty:
                print("Query returned no results")
                return
            
            print(f"Query returned {len(df)} rows and {len(df.columns)} columns")
            print("\nFirst few rows:")
            print(df.head())
            
            # Auto-detect plot type if not specified
            if plot_type == 'auto':
                plot_type = self._auto_detect_plot_type(df)
            
            # Create appropriate plot
            if plot_type == 'line':
                self._plot_line_data(df, figsize)
            elif plot_type == 'bar':
                self._plot_bar_data(df, figsize)
            elif plot_type == 'scatter':
                self._plot_scatter_data(df, figsize)
            elif plot_type == 'histogram':
                self._plot_histogram_data(df, figsize)
            else:
                print(f"Unknown plot type: {plot_type}")
                
        except Exception as e:
            print(f"Error executing query: {e}")
    
    def _auto_detect_plot_type(self, df: pd.DataFrame) -> str:
        """
        Auto-detect the best plot type for the data
        """
        if len(df.columns) >= 2:
            # Check if there's a date column
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                return 'line'
            
            # Check if data looks like categorical
            if df.iloc[:, 0].dtype == 'object' and len(df) <= 20:
                return 'bar'
            
            # Check if data looks like scatter
            if df.iloc[:, 0].dtype in ['float64', 'int64'] and df.iloc[:, 1].dtype in ['float64', 'int64']:
                return 'scatter'
        
        # Default to histogram for single column
        return 'histogram'
    
    def _plot_line_data(self, df: pd.DataFrame, figsize: Tuple[int, int]):
        """Plot line chart"""
        fig, ax = plt.subplots(figsize=figsize)
        
        if len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = df.columns[1]
            
            if df[x_col].dtype == 'object':
                # Try to convert to datetime
                try:
                    df[x_col] = pd.to_datetime(df[x_col])
                except:
                    pass
            
            ax.plot(df[x_col], df[y_col], marker='o', linewidth=2, markersize=4)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'{y_col} over {x_col}', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def _plot_bar_data(self, df: pd.DataFrame, figsize: Tuple[int, int]):
        """Plot bar chart"""
        fig, ax = plt.subplots(figsize=figsize)
        
        if len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = df.columns[1]
            
            ax.bar(df[x_col], df[y_col], color=self.colors['default'])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'{y_col} by {x_col}', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def _plot_scatter_data(self, df: pd.DataFrame, figsize: Tuple[int, int]):
        """Plot scatter chart"""
        fig, ax = plt.subplots(figsize=figsize)
        
        if len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = df.columns[1]
            
            ax.scatter(df[x_col], df[y_col], alpha=0.6, color=self.colors['default'])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'{y_col} vs {x_col}', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _plot_histogram_data(self, df: pd.DataFrame, figsize: Tuple[int, int]):
        """Plot histogram"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use the first numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]
            ax.hist(df[col], bins=30, alpha=0.7, color=self.colors['default'])
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {col}', fontweight='bold')
        else:
            # Use the first column
            col = df.columns[0]
            ax.hist(df[col], bins=30, alpha=0.7, color=self.colors['default'])
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {col}', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_training_data_analysis(self, train_file: str = "train_data.json", 
                                   figsize: Tuple[int, int] = (15, 10)):
        """
        Analyze and visualize training data
        """
        import json
        
        try:
            with open(train_file, 'r') as f:
                train_data = json.load(f)
            
            # Extract queries and analyze patterns
            queries = []
            sql_lengths = []
            query_types = []
            
            for item in train_data:
                query = item['input'].split('Query: ')[1]
                queries.append(query)
                sql_lengths.append(len(item['output']))
                
                # Categorize query types
                query_lower = query.lower()
                if 'volatility' in query_lower or 'vol' in query_lower:
                    query_types.append('Volatility')
                elif 'correlation' in query_lower:
                    query_types.append('Correlation')
                elif 'compare' in query_lower or 'vs' in query_lower:
                    query_types.append('Comparison')
                elif 'best' in query_lower or 'top' in query_lower:
                    query_types.append('Ranking')
                elif 'average' in query_lower or 'avg' in query_lower:
                    query_types.append('Aggregation')
                else:
                    query_types.append('Basic')
            
            # Create analysis plots
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # Plot 1: Query type distribution
            type_counts = pd.Series(query_types).value_counts()
            axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Query Type Distribution', fontweight='bold')
            
            # Plot 2: SQL length distribution
            axes[0, 1].hist(sql_lengths, bins=30, alpha=0.7, color=self.colors['default'])
            axes[0, 1].set_xlabel('SQL Length (characters)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('SQL Query Length Distribution', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Stock mentions
            stock_mentions = {}
            for query in queries:
                for stock in ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NFLX', 'SPY']:
                    if stock in query:
                        stock_mentions[stock] = stock_mentions.get(stock, 0) + 1
            
            if stock_mentions:
                stocks = list(stock_mentions.keys())
                counts = list(stock_mentions.values())
                axes[1, 0].bar(stocks, counts, color=[self.colors.get(s, self.colors['default']) for s in stocks])
                axes[1, 0].set_xlabel('Stock')
                axes[1, 0].set_ylabel('Mentions')
                axes[1, 0].set_title('Stock Mentions in Training Data', fontweight='bold')
                plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
            
            # Plot 4: Query length distribution
            query_lengths = [len(q) for q in queries]
            axes[1, 1].hist(query_lengths, bins=30, alpha=0.7, color=self.colors['default'])
            axes[1, 1].set_xlabel('Query Length (characters)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Natural Language Query Length Distribution', fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Print summary statistics
            print(f"\nTraining Data Analysis Summary:")
            print(f"Total queries: {len(queries)}")
            print(f"Average SQL length: {np.mean(sql_lengths):.1f} characters")
            print(f"Average query length: {np.mean(query_lengths):.1f} characters")
            print(f"Query types: {type_counts.to_dict()}")
            
        except FileNotFoundError:
            print(f"Training data file {train_file} not found")
        except Exception as e:
            print(f"Error analyzing training data: {e}")

def main():
    """
    Demo function to showcase visualization capabilities
    """
    print("📊 Financial Data Visualization Demo")
    print("=" * 50)
    
    viz = FinancialDataVisualizer()
    
    # Demo 1: Price history
    print("\n1. Plotting price history...")
    viz.plot_price_history(['AAPL', 'TSLA', 'GOOGL'], days=90)
    
    # Demo 2: Returns distribution
    print("\n2. Plotting returns distribution...")
    viz.plot_returns_distribution(['AAPL', 'TSLA', 'GOOGL', 'MSFT'], days=365)
    
    # Demo 3: Volatility comparison
    print("\n3. Plotting volatility comparison...")
    viz.plot_volatility_comparison(['AAPL', 'TSLA', 'GOOGL', 'MSFT'], days=365)
    
    # Demo 4: Correlation heatmap
    print("\n4. Plotting correlation heatmap...")
    viz.plot_correlation_heatmap(['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN'], days=365)
    
    # Demo 5: Volume analysis
    print("\n5. Plotting volume analysis...")
    viz.plot_volume_analysis('AAPL', days=90)
    
    # Demo 6: SQL query results
    print("\n6. Plotting SQL query results...")
    sql_query = "SELECT Date, Close FROM stock_data WHERE ticker = 'AAPL' AND Date >= date('now', '-30 days') ORDER BY Date"
    viz.plot_sql_query_results(sql_query, plot_type='line')
    
    # Demo 7: Training data analysis (if available)
    print("\n7. Analyzing training data...")
    viz.plot_training_data_analysis()
    
    print("\n✅ Visualization demo complete!")

if __name__ == "__main__":
    main() 