import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_market_calendars import get_calendar
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UNGTemperatureAnalyzer:
    """Analyzer for UNG stock price and temperature relationship"""
    
    def __init__(self, stock_file, temp_file, storage_file=None, use_storage_release_date=True):
        self.stock_file = stock_file
        self.temp_file = temp_file
        self.storage_file = storage_file
        self.use_storage_release_date = use_storage_release_date
        self.storage_data = None
        self.stock_data = None
        self.temp_data = None
        self.merged_data = None
        
        # Configuration
        self.config = {
            'analysis_period': {'start_day': 330, 'end_day': 420},
            'temperature_lags': [1, 3, 5, 7],
            'rolling_windows': [3, 5, 7, 30],
            'output_file': 'UNG_Temperature_Analysis_Results.csv'
        }
    
    def load_stock_data(self):
        """Load and preprocess UNG stock data"""
        try:
            logger.info(f"Loading stock data from {self.stock_file}")
            df = pd.read_csv(self.stock_file)
            
            # Data validation
            if df.empty:
                raise ValueError("Stock data file is empty")
                
            # Preprocessing
            df['Date'] = pd.to_datetime(df['Date'])
            df['Year'] = df['Date'].dt.year
            df['Day_of_Year'] = df['Date'].dt.dayofyear
            df['Month'] = df['Date'].dt.strftime('%b')
            df['Month_Day'] = df['Date'].dt.strftime('%m-%d')
            
            # Create day count from 2023-01-01
            start_date = pd.Timestamp("2023-01-01")
            df['Day_Count'] = (df['Date'] - start_date).dt.days + 1
            
            logger.info(f"Stock data loaded: {len(df)} rows, {df['Year'].nunique()} years")
            return df
            
        except Exception as e:
            logger.error(f"Error loading stock data: {e}")
            raise
    
    
    def load_storage_data(self):
        """Load and preprocess EIA weekly natural gas storage data.

        Expected columns: date, storage_bcf, weekly_change_bcf
        The 'date' in the provided EIA file appears to be the week-ending date (typically Friday).
        For trading-day alignment, we optionally approximate a release date as the following Thursday
        (week-ending Friday + 6 days). This is a pragmatic approximation for exploratory analysis.
        """
        if not self.storage_file:
            raise ValueError("storage_file not provided")

        try:
            df = pd.read_csv(self.storage_file)
            if df.empty:
                raise ValueError("Storage data file is empty")

            required_cols = {"date", "storage_bcf", "weekly_change_bcf"}
            missing = required_cols - set(df.columns)
            if missing:
                raise ValueError(f"Storage data missing required columns: {sorted(missing)}")

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            # Approximate report release date: Thursday after week-ending Friday
            df["release_date"] = df["date"] + pd.Timedelta(days=6)

            logger.info(f"Storage data loaded: {len(df)} rows")
            return df

        except Exception as e:
            logger.error(f"Error loading storage data: {e}")
            raise

    def merge_storage_to_trading_days(self, merged_df: pd.DataFrame, storage_df: pd.DataFrame):
        """Attach the most recent weekly storage value available as-of each trading day.

        Uses pandas.merge_asof:
          - left: trading days (merged_df index)
          - right: weekly storage keyed by either 'release_date' or 'date'
        """
        if merged_df is None or merged_df.empty:
            raise ValueError("merged_df is empty; run merge_datasets() first")

        key = "release_date" if self.use_storage_release_date else "date"

        left = merged_df.reset_index().rename(columns={"index": "Date"}).sort_values("Date")
        right = storage_df[[key, "storage_bcf", "weekly_change_bcf"]].dropna(subset=[key]).sort_values(key)

        out = pd.merge_asof(
            left,
            right,
            left_on="Date",
            right_on=key,
            direction="backward",
            allow_exact_matches=True
        )

        # Optional convenience feature: weekly change as percent of storage
        out["storage_change_pct"] = out["weekly_change_bcf"] / out["storage_bcf"]

        # Drop the merge key column from the right side to keep the frame tidy
        if key in out.columns:
            out = out.drop(columns=[key])

        out = out.set_index("Date")
        logger.info("Storage data merged to trading days (as-of join)")
        return out

    def load_temperature_data(self):
        """Load and preprocess temperature data"""
        try:
            logger.info(f"Loading temperature data from {self.temp_file}")
            df = pd.read_csv(self.temp_file)
            
            if df.empty:
                raise ValueError("Temperature data file is empty")
                
            # Preprocessing
            df['Date'] = pd.to_datetime(df['Date'], format="%Y%m%d")
            df['Year'] = df['Date'].dt.year
            df['Day_of_Year'] = df['Date'].dt.dayofyear
            
            # Create day count from 2023-01-01
            start_date = pd.Timestamp("2023-01-01")
            df['Day_Count'] = (df['Date'] - start_date).dt.days + 1
            
            logger.info(f"Temperature data loaded: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading temperature data: {e}")
            raise


    def _get_temperature_analysis_window(self):
        """Return (start_date, end_date) implied by the temperature dataset, if available.

        This is used to ensure storage analysis is run over the same window as temperature analysis.
        """
        try:
            if not getattr(self, "temp_file", None):
                return None, None
            tmp = self.load_temperature_data()
            if tmp is None or tmp.empty:
                return None, None
            idx = tmp.index
            # idx may be datetime or int Day_Count; if int, cannot infer dates reliably here
            if not hasattr(idx, "min") or not hasattr(idx, "max"):
                return None, None
            if isinstance(idx.min(), (int, float)):
                return None, None
            start = pd.to_datetime(idx.min())
            end = pd.to_datetime(idx.max())
            return start, end
        except Exception:
            return None, None

    
    def merge_datasets(self):
        """Merge stock and temperature data with proper date alignment"""
        try:
            # Filter data for analysis period
            start_day = self.config['analysis_period']['start_day']
            end_day = self.config['analysis_period']['end_day']
            
            stock_filtered = self.stock_data[
                (self.stock_data['Day_Count'] >= start_day) & 
                (self.stock_data['Day_Count'] <= end_day)
            ].copy()
            
            temp_filtered = self.temp_data[
                (self.temp_data['Day_Count'] >= start_day) & 
                (self.temp_data['Day_Count'] <= end_day)
            ].copy()
            
            # Set date indices
            stock_filtered.set_index('Date', inplace=True)
            temp_filtered.set_index('Date', inplace=True)
            
            # Get NYSE trading calendar
            nyse = get_calendar('NYSE')
            trading_days = nyse.valid_days(
                start_date=temp_filtered.index.min(),
                end_date=temp_filtered.index.max()
            ).tz_localize(None)
            
            # Create full temperature timeline and align with trading days
            full_temp = temp_filtered.reindex(
                pd.date_range(
                    start=temp_filtered.index.min(),
                    end=temp_filtered.index.max(),
                    freq='D'
                )
            )
            full_temp.ffill(inplace=True)  # Forward fill missing values
            aligned_temp = full_temp[full_temp.index.isin(trading_days)]
            
            # Merge datasets
            merged_df = stock_filtered.join(aligned_temp, how='left', rsuffix='_temp')
            
            # Handle any remaining missing values
            merged_df['tm000'].interpolate(method='time', inplace=True)
            
            logger.info(f"Merged dataset created: {len(merged_df)} rows")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error merging datasets: {e}")
            raise
    
    def create_temperature_features(self, df):
        """Create comprehensive temperature-based features"""
        logger.info("Creating temperature features")
        
        # Convert temperature from Kelvin to Celsius for interpretation
        df['temperature_c'] = df['tm000'] - 273.15
        
        # Temperature lag features
        for lag in self.config['temperature_lags']:
            df[f'temp_lag_{lag}d'] = df['tm000'].shift(lag)
            df[f'temp_c_lag_{lag}d'] = df['temperature_c'].shift(lag)
        
        # Rolling statistics
        for window in self.config['rolling_windows']:
            df[f'temp_ma_{window}d'] = df['tm000'].rolling(window).mean()
            df[f'temp_std_{window}d'] = df['tm000'].rolling(window).std()
        
        # Temperature anomaly (deviation from 30-day average)
        df['temp_anomaly'] = df['tm000'] - df['temp_ma_30d']
        df['temp_anomaly_c'] = df['temperature_c'] - (df['temp_ma_30d'] - 273.15)
        
        # Temperature volatility
        df['temp_volatility_5d'] = df['tm000'].rolling(5).std()
        df['temp_volatility_10d'] = df['tm000'].rolling(10).std()
        
        # Temperature changes
        df['temp_change_1d'] = df['tm000'].diff()
        df['temp_change_pct_1d'] = df['tm000'].pct_change()
        
        logger.info("Temperature features created successfully")
        return df
    
    def create_stock_features(self, df):
        """Create stock price-based features"""
        logger.info("Creating stock features")
        
        # Price changes
        df['price_change_1d'] = df['Close'].diff()
        df['price_change_pct_1d'] = df['Close'].pct_change()
        
        # Volume features
        df['volume_ma_5d'] = df['Volume'].rolling(5).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma_5d']
        
        # Price ranges
        df['daily_range'] = df['High'] - df['Low']
        df['range_pct'] = df['daily_range'] / df['Close']
        
        logger.info("Stock features created successfully")
        return df
    
    def calculate_r_squared(self, df):
        """Calculate R-squared values for temperature features vs Open price"""
        logger.info("Calculating R-squared values")
        
        temperature_features = [
            'temperature_c',
            'temp_c_lag_1d', 'temp_c_lag_3d', 'temp_c_lag_5d', 'temp_c_lag_7d',
            'temp_anomaly_c',
            'temp_volatility_5d'
        ]
        
        r_squared_results = []
        
        for feature in temperature_features:
            if feature in df.columns:
                # Clean data - remove rows where either variable is NaN
                clean_data = df[['Open', feature]].dropna()
                
                if len(clean_data) > 1:  # Need at least 2 points
                    X = clean_data[feature].values.reshape(-1, 1)
                    y = clean_data['Open'].values
                    
                    # Calculate R-squared using linear regression
                    model = LinearRegression()
                    model.fit(X, y)
                    y_pred = model.predict(X)
                    r2 = r2_score(y, y_pred)
                    
                    # Also calculate correlation for reference
                    correlation, p_value = pearsonr(clean_data['Open'], clean_data[feature])
                    
                    r_squared_results.append({
                        'feature': feature,
                        'r_squared': r2,
                        'correlation': correlation,
                        'p_value': p_value,
                        'sample_size': len(clean_data),
                        'coefficient': model.coef_[0],
                        'intercept': model.intercept_
                    })
        
        r_squared_df = pd.DataFrame(r_squared_results)
        
        # Sort by absolute R-squared value (descending)
        r_squared_df = r_squared_df.reindex(
            r_squared_df['r_squared'].abs().sort_values(ascending=False).index
        )
        
        logger.info("R-squared calculations completed")
        return r_squared_df
    
    def plot_r_squared_comparison(self, r_squared_df):
        """Plot R-squared values for different temperature features"""
        logger.info("Creating R-squared comparison plot")
        
        plt.figure(figsize=(12, 8))
        
        # Create bar plot
        bars = plt.barh(r_squared_df['feature'], r_squared_df['r_squared'], 
                       color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for i, (bar, r2) in enumerate(zip(bars, r_squared_df['r_squared'])):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{r2:.3f}', va='center', ha='left', fontweight='bold')
        
        plt.xlabel('R-squared Value', fontsize=12, fontweight='bold')
        plt.ylabel('Temperature Features', fontsize=12, fontweight='bold')
        plt.title('R-squared: Temperature Features vs UNG Open Price', 
                 fontsize=14, fontweight='bold')
        
        # Add reference lines
        plt.axvline(x=0, color='black', linewidth=0.8)
        plt.axvline(x=0.1, color='red', linestyle='--', alpha=0.5, 
                   label='R² = 0.1 (10% variance explained)')
        plt.axvline(x=0.25, color='orange', linestyle='--', alpha=0.5,
                   label='R² = 0.25 (25% variance explained)')
        plt.axvline(x=0.5, color='green', linestyle='--', alpha=0.5,
                   label='R² = 0.5 (50% variance explained)')
        
        plt.grid(True, alpha=0.3, axis='x')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig('r_squared_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("R-squared comparison plot saved")
    
    def plot_best_r_squared_relationship(self, df, best_feature, r_squared_value):
        """Plot the relationship for the feature with highest R-squared"""
        logger.info(f"Creating detailed plot for best feature: {best_feature}")
        
        plt.figure(figsize=(10, 6))
        
        # Clean data
        clean_data = df[['Open', best_feature]].dropna()
        
        # Create scatter plot
        plt.scatter(clean_data[best_feature], clean_data['Open'], 
                   alpha=0.6, s=50, color='blue', edgecolors='black', linewidth=0.5)
        
        # Add regression line
        X = clean_data[best_feature].values.reshape(-1, 1)
        y = clean_data['Open'].values
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        plt.plot(clean_data[best_feature], y_pred, color='red', linewidth=2, 
                label=f'Regression Line (R² = {r_squared_value:.3f})')
        
        plt.xlabel(f'{best_feature} (°C)', fontsize=12, fontweight='bold')
        plt.ylabel('UNG Open Price ($)', fontsize=12, fontweight='bold')
        plt.title(f'UNG Open Price vs {best_feature}\nR-squared = {r_squared_value:.3f}', 
                 fontsize=14, fontweight='bold')
        
        # Add equation to plot
        equation = f'y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}'
        plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', 
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'best_r_squared_relationship_{best_feature}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Best R-squared relationship plot saved for {best_feature}")
    
    def plot_open_price_vs_temperature(self, df):
        """Plot 1: Open Price vs Temperature (5-day lag)"""
        logger.info("Creating Open Price vs Temperature plot")
        
        plt.figure(figsize=(12, 6))
        
        # Primary y-axis for Open Price
        ax1 = plt.gca()
        color1 = 'tab:blue'
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Open Price ($)', color=color1, fontsize=12, fontweight='bold')
        line1 = ax1.plot(df.index, df['Open'], label='Open Price', 
                        color=color1, linewidth=2, alpha=0.8)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Secondary y-axis for Temperature
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Temperature Lag 5d (°C)', color=color2, fontsize=12, fontweight='bold')
        line2 = ax2.plot(df.index, df['temp_c_lag_5d'], label='Temperature Lag 5d', 
                        color=color2, linewidth=2, alpha=0.7, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        plt.title('UNG Open Price vs Temperature (5-day Lag)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save individual plot
        plt.savefig('open_price_vs_temperature.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Open Price vs Temperature plot saved")
    
    def plot_correlation_heatmap(self, df):
        """Plot 2: Correlation heatmap between prices and temperature lags"""
        logger.info("Creating correlation heatmap")
        
        plt.figure(figsize=(10, 8))
        
        # Select features for correlation
        correlation_features = ['Open', 'Close', 'High', 'Low'] + \
                             [f'temp_c_lag_{lag}d' for lag in [1, 3, 5, 7]]
        
        # Calculate correlation matrix
        corr_data = df[correlation_features].corr()
        price_temp_corr = corr_data.loc[['Open', 'Close', 'High', 'Low'], 
                                      [f'temp_c_lag_{lag}d' for lag in [1, 3, 5, 7]]]
        
        # Create heatmap
        sns.heatmap(price_temp_corr, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   vmin=-1, 
                   vmax=1,
                   square=True,
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title('Price vs Temperature Lag Correlation Matrix', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save individual plot
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Correlation heatmap saved")
    
    def plot_temperature_anomaly_vs_price_changes(self, df):
        """Plot 3: Temperature anomaly vs price changes scatter plot"""
        logger.info("Creating temperature anomaly vs price changes plot")
        
        # Create a clean dataset with no missing values in either column
        clean_data = df[['temp_anomaly_c', 'price_change_pct_1d', 'Volume']].dropna()
        
        if len(clean_data) == 0:
            logger.warning("No data available for temperature anomaly vs price changes plot")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot using clean data
        scatter = plt.scatter(clean_data['temp_anomaly_c'], 
                            clean_data['price_change_pct_1d'] * 100, 
                            c=clean_data['Volume'], 
                            alpha=0.6, 
                            cmap='viridis',
                            s=50)
        
        plt.xlabel('Temperature Anomaly (°C)', fontsize=12)
        plt.ylabel('Daily Price Change (%)', fontsize=12)
        plt.title('Temperature Anomaly vs Price Changes', fontsize=14, fontweight='bold')
        
        # Add reference lines
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Volume')
        
        # Add trend line - Use the clean data that has same length for x and y
        if len(clean_data) > 1:  # Need at least 2 points for trend line
            try:
                z = np.polyfit(clean_data['temp_anomaly_c'], 
                              clean_data['price_change_pct_1d'] * 100, 1)
                p = np.poly1d(z)
                plt.plot(clean_data['temp_anomaly_c'], p(clean_data['temp_anomaly_c']), 
                        "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.3f})')
                plt.legend()
            except Exception as e:
                logger.warning(f"Could not create trend line: {e}")
        
        plt.tight_layout()
        
        # Save individual plot
        plt.savefig('temperature_anomaly_vs_price_changes.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Temperature anomaly vs price changes plot saved with {len(clean_data)} data points")
    
    def plot_volume_vs_temperature_volatility(self, df):
        """Plot 4: Volume vs temperature volatility"""
        logger.info("Creating volume vs temperature volatility plot")
        
        # Ensure we have data for both series
        clean_data = df[['Volume', 'temp_volatility_5d']].dropna()
        
        if len(clean_data) == 0:
            logger.warning("No data available for volume vs temperature volatility plot")
            return
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot volume on primary y-axis
        color1 = 'green'
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Volume (Millions)', color=color1, fontsize=12, fontweight='bold')
        ax1.plot(clean_data.index, clean_data['Volume'] / 1e6, 
                label='Volume (Millions)', 
                color=color1, 
                alpha=0.7,
                linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Plot temperature volatility on secondary y-axis
        ax2 = ax1.twinx()
        color2 = 'orange'
        ax2.set_ylabel('Temperature Volatility (5-day STD)', color=color2, fontsize=12, fontweight='bold')
        ax2.plot(clean_data.index, clean_data['temp_volatility_5d'], 
                label='Temperature Volatility (5d)', 
                color=color2, 
                alpha=0.7, 
                linestyle='--',
                linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.title('Volume vs Temperature Volatility', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save individual plot
        plt.savefig('volume_vs_temperature_volatility.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Volume vs temperature volatility plot saved")
    
    def analyze_correlations(self, df):
        """Perform comprehensive correlation analysis"""
        logger.info("Performing correlation analysis")
        
        # Select features for correlation analysis
        price_features = ['Open', 'Close', 'High', 'Low']
        temp_features = [f'temp_lag_{lag}d' for lag in self.config['temperature_lags']]
        temp_features.extend(['temp_anomaly', 'temp_volatility_5d'])
        
        analysis_features = price_features + temp_features
        
        # Calculate correlation matrix
        corr_matrix = df[analysis_features].corr()
        
        # Calculate statistical significance
        significant_correlations = []
        for price_col in price_features:
            for temp_col in temp_features:
                if temp_col in df.columns:
                    # Use dropna on both columns together to ensure same length
                    valid_data = df[[price_col, temp_col]].dropna()
                    if len(valid_data) > 2:
                        try:
                            corr, p_value = pearsonr(valid_data[price_col], valid_data[temp_col])
                            significant_correlations.append({
                                'price_feature': price_col,
                                'temp_feature': temp_col,
                                'correlation': corr,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            })
                        except Exception as e:
                            logger.warning(f"Could not calculate correlation between {price_col} and {temp_col}: {e}")
        
        correlation_df = pd.DataFrame(significant_correlations)
        
        logger.info("Correlation analysis completed")
        return corr_matrix, correlation_df
    
    def generate_report(self, df, corr_matrix, significant_correlations, r_squared_df):
        """Generate analysis report with R-squared results"""
        logger.info("Generating analysis report")
        
        print("=" * 70)
        print("UNG STOCK & TEMPERATURE ANALYSIS REPORT")
        print("=" * 70)
        
        print(f"\nDATA OVERVIEW:")
        print(f"Analysis period: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        print(f"Total trading days: {len(df)}")
        print(f"Average Open Price: ${df['Open'].mean():.2f}")
        print(f"Average Temperature: {df['temperature_c'].mean():.1f}°C")
        
        print(f"\nR-SQUARED ANALYSIS (Temperature vs Open Price):")
        print("-" * 50)
        if len(r_squared_df) > 0:
            for _, row in r_squared_df.iterrows():
                significance = "***" if row['p_value'] < 0.05 else ""
                print(f"{row['feature']:20} | R² = {row['r_squared']:6.3f} | "
                      f"Corr = {row['correlation']:6.3f} | p = {row['p_value']:6.3f} {significance}")
            
            best_feature = r_squared_df.iloc[0]
            print(f"\nBEST PREDICTOR: {best_feature['feature']}")
            print(f"R-squared: {best_feature['r_squared']:.3f} "
                  f"({best_feature['r_squared']*100:.1f}% of variance explained)")
            print(f"Regression: Open Price = {best_feature['coefficient']:.3f} × {best_feature['feature']} "
                  f"+ {best_feature['intercept']:.3f}")
        else:
            print("No R-squared results available")
        
        print(f"\nTOP CORRELATIONS:")
        print("-" * 50)
        if len(significant_correlations) > 0:
            significant_correlations = significant_correlations.sort_values('correlation', key=abs, ascending=False)
            for _, row in significant_correlations.head(10).iterrows():
                significance = "***" if row['significant'] else ""
                print(f"{row['price_feature']} vs {row['temp_feature']}: "
                      f"{row['correlation']:.3f} (p={row['p_value']:.3f}){significance}")
        else:
            print("No significant correlations found")
        
        print(f"\nKEY INSIGHTS:")
        if len(r_squared_df) > 0:
            best_feature = r_squared_df.iloc[0]
            print(f"• Temperature explains {best_feature['r_squared']*100:.1f}% of UNG price variance")
            print(f"• Best predictor: {best_feature['feature']}")
            
            if best_feature['r_squared'] > 0.5:
                print("• STRONG relationship: Temperature is a major factor")
            elif best_feature['r_squared'] > 0.25:
                print("• MODERATE relationship: Temperature has noticeable impact")
            elif best_feature['r_squared'] > 0.1:
                print("• WEAK relationship: Temperature has minor influence")
            else:
                print("• VERY WEAK relationship: Temperature has little predictive power")
        else:
            print("• No strong temperature-price relationship detected")
        
        if len(df) > 30:
            recent_trend = "increasing" if df['Open'].iloc[-1] > df['Open'].iloc[-30] else "decreasing"
            temp_trend = "warmer" if df['temperature_c'].iloc[-1] > df['temperature_c'].iloc[-30] else "cooler"
            print(f"• Recent price trend: {recent_trend}")
            print(f"• Recent temperature trend: {temp_trend}")
        
        print("=" * 70)
    
    def save_results(self, df, r_squared_df):
        """Save analysis results to file"""
        try:
            # Save main data
            output_file = self.config['output_file']
            df.to_csv(output_file)
            logger.info(f"Analysis results saved to {output_file}")
            
            # Save R-squared results
            r_squared_file = 'r_squared_results.csv'
            r_squared_df.to_csv(r_squared_file, index=False)
            logger.info(f"R-squared results saved to {r_squared_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def run_analysis_temperature(self):
        """Main method to run complete analysis"""
        try:
            logger.info("Starting UNG Temperature Analysis")
            
            # Load data
            self.stock_data = self.load_stock_data()
            self.temp_data = self.load_temperature_data()
            
            # Merge datasets
            self.merged_data = self.merge_datasets()

            # Persist analysis window (used to keep storage analysis on the same period)
            self.analysis_start = self.merged_data.index.min()
            self.analysis_end = self.merged_data.index.max()

            
            # Create features
            self.merged_data = self.create_temperature_features(self.merged_data)
            self.merged_data = self.create_stock_features(self.merged_data)
            
            # Calculate R-squared values
            r_squared_df = self.calculate_r_squared(self.merged_data)
            
            # Analyze correlations
            corr_matrix, significant_correlations = self.analyze_correlations(self.merged_data)
            
            # Generate R-squared plots
            if len(r_squared_df) > 0:
                self.plot_r_squared_comparison(r_squared_df)
                best_feature = r_squared_df.iloc[0]
                self.plot_best_r_squared_relationship(
                    self.merged_data, 
                    best_feature['feature'], 
                    best_feature['r_squared']
                )
            
            # Generate individual plots
            self.plot_open_price_vs_temperature(self.merged_data)
            self.plot_correlation_heatmap(self.merged_data)
            self.plot_temperature_anomaly_vs_price_changes(self.merged_data)
            self.plot_volume_vs_temperature_volatility(self.merged_data)
            
            # Generate report
            self.generate_report(self.merged_data, corr_matrix, significant_correlations, r_squared_df)
            
            # Save results
            self.save_results(self.merged_data, r_squared_df)
            
            logger.info("Analysis completed successfully")
            return self.merged_data, r_squared_df
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise


    def build_daily_storage_series(self, stock_df: pd.DataFrame, storage_df: pd.DataFrame) -> pd.DataFrame:
        """Convert weekly EIA storage observations into a daily (trading-day) series aligned to stock_df's index.

        The resulting series is piecewise-constant: each trading day is assigned the most recent storage value
        that would have been known as-of that day (using either 'release_date' or 'date' as the effective date).
        """
        if stock_df is None or stock_df.empty:
            raise ValueError("stock_df is empty; load stock data first")
        if storage_df is None or storage_df.empty:
            raise ValueError("storage_df is empty; load storage data first")

        key = "release_date" if self.use_storage_release_date else "date"
        if key not in storage_df.columns:
            raise ValueError(f"Storage data missing '{key}' column; check load_storage_data()")

        # Build trading-day index from stock data
        trading_index = pd.DatetimeIndex(stock_df.index).sort_values()

        weekly = (
            storage_df[[key, "storage_bcf", "weekly_change_bcf"]]
            .dropna(subset=[key])
            .sort_values(key)
            .set_index(key)
        )

        # Reindex onto trading days and forward-fill to represent "latest known" storage level.
        daily = weekly.reindex(trading_index).ffill()
        daily["storage_change_pct"] = daily["weekly_change_bcf"] / daily["storage_bcf"]
        return daily

    def run_analysis_storage(self):
        """Run storage-focused analysis: build daily storage series and analyze relationship to UNG.

        Outputs:
          - Correlations (levels and returns)
          - R-squared (simple linear models)
          - Plots
          - CSV results (separate from temperature output)
        """
        try:
            logger.info("Starting UNG Storage Analysis")

            # Load data (stock + storage only)
            self.stock_data = self.load_stock_data()
            self.storage_data = self.load_storage_data()

            # Align analysis window to temperature analysis period when available
            start = getattr(self, 'analysis_start', None)
            end = getattr(self, 'analysis_end', None)
            if start is None or end is None:
                tstart, tend = self._get_temperature_analysis_window()
                start = tstart if start is None else start
                end = tend if end is None else end
            if start is not None and end is not None:
                #self.stock_data = self.stock_data.loc[(self.stock_data.index >= start) & (self.stock_data.index <= end)].copy()
                self.stock_data["Date"] = pd.to_datetime(self.stock_data["Date"])
                self.stock_data = self.stock_data.loc[ (self.stock_data["Date"] >= start) & (self.stock_data["Date"] <= end) ].copy()

            # Prepare stock index
            stock = self.stock_data.copy().set_index("Date").sort_index()

            # Build daily storage aligned to trading days
            daily_storage = self.build_daily_storage_series(stock, self.storage_data)

            # Combine for analysis
            df = stock.join(daily_storage, how="left").dropna(subset=["storage_bcf"]).copy()

            # Basic derived metrics
            df["Close_Return"] = df["Close"].pct_change()
            df["Open_Return"] = df["Open"].pct_change()
            df["Storage_Change"] = df["weekly_change_bcf"]  # piecewise constant within a week

            # Correlation matrix (focus on interpretable variables)
            corr_cols = ["Close", "Open", "Volume", "storage_bcf", "weekly_change_bcf", "storage_change_pct", "Close_Return"]
            corr_cols = [c for c in corr_cols if c in df.columns]
            corr_matrix = df[corr_cols].corr()

            # Simple R^2 models (levels and returns)
            r2_rows = []

            def _r2(y, x):
                data = df[[y, x]].dropna()
                if len(data) < 20:
                    return np.nan
                X = data[[x]].values
                yv = data[y].values
                model = LinearRegression().fit(X, yv)
                yhat = model.predict(X)
                return r2_score(yv, yhat)

            # Level: Close ~ storage_bcf
            if "Close" in df.columns and "storage_bcf" in df.columns:
                r2_rows.append({"Model": "Close ~ storage_bcf", "R_squared": _r2("Close", "storage_bcf")})

            # Return: Close_Return ~ weekly_change_bcf (note: weekly_change is flat intra-week; still useful as a baseline)
            if "Close_Return" in df.columns and "weekly_change_bcf" in df.columns:
                r2_rows.append({"Model": "Close_Return ~ weekly_change_bcf", "R_squared": _r2("Close_Return", "weekly_change_bcf")})

            r2_df = pd.DataFrame(r2_rows)

            # Save results
            out_corr = "UNG_Storage_Correlation_Matrix.csv"
            out_r2 = "UNG_Storage_R_Squared.csv"
            corr_matrix.to_csv(out_corr)
            r2_df.to_csv(out_r2, index=False)
            logger.info(f"Storage correlation matrix saved to {out_corr}")
            logger.info(f"Storage R-squared results saved to {out_r2}")

            # Plots
            self.plot_price_vs_storage(df)
            self.plot_storage_time_series(df)
            self.plot_storage_correlation_heatmap(corr_matrix)
            self.plot_weekly_change_vs_returns_event_window(stock, self.storage_data)

            logger.info("UNG Storage Analysis completed successfully")
            return df, corr_matrix, r2_df

        except Exception as e:
            logger.error(f"Error in storage analysis: {e}")
            raise

    def plot_price_vs_storage(self, df: pd.DataFrame):
        """Scatter: UNG Close vs storage level."""
        try:
            plt.figure(figsize=(10, 6))
            x = df["storage_bcf"]
            y = df["Close"]
            plt.scatter(x, y, alpha=0.5)
            plt.xlabel("EIA Working Gas in Storage (Bcf)")
            plt.ylabel("UNG Close Price")
            plt.title("UNG Close Price vs EIA Natural Gas Storage")

            # Regression line
            data = df[["storage_bcf", "Close"]].dropna()
            if len(data) >= 20:
                X = data[["storage_bcf"]].values
                yv = data["Close"].values
                model = LinearRegression().fit(X, yv)
                xline = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                yline = model.predict(xline)
                plt.plot(xline, yline, linewidth=2)

            plt.tight_layout()
            plt.savefig("UNG_Close_vs_Storage.png", dpi=300)
            plt.close()
            logger.info("Saved plot: UNG_Close_vs_Storage.png")
        except Exception as e:
            logger.error(f"Error plotting price vs storage: {e}")

    def plot_storage_time_series(self, df: pd.DataFrame):
        """Time series: UNG Close and storage level (two axes)."""
        try:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(df.index, df["Close"])
            ax1.set_xlabel("Date")
            ax1.set_ylabel("UNG Close Price")

            ax2 = ax1.twinx()
            ax2.plot(df.index, df["storage_bcf"])
            ax2.set_ylabel("Storage (Bcf)")

            plt.title("UNG Close Price and EIA Storage Over Time")
            fig.tight_layout()
            plt.savefig("UNG_Close_and_Storage_TimeSeries.png", dpi=300)
            plt.close()
            logger.info("Saved plot: UNG_Close_and_Storage_TimeSeries.png")
        except Exception as e:
            logger.error(f"Error plotting storage time series: {e}")

    def plot_storage_correlation_heatmap(self, corr_matrix: pd.DataFrame):
        """Heatmap for storage-related correlation matrix."""
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
            plt.title("Correlation Matrix: UNG and EIA Storage Variables")
            plt.tight_layout()
            plt.savefig("UNG_Storage_Correlation_Heatmap.png", dpi=300)
            plt.close()
            logger.info("Saved plot: UNG_Storage_Correlation_Heatmap.png")
        except Exception as e:
            logger.error(f"Error plotting storage correlation heatmap: {e}")

    def plot_weekly_change_vs_returns_event_window(self, stock_df: pd.DataFrame, storage_df: pd.DataFrame, window_days: int = 1):
        """Event-style scatter: weekly storage change vs UNG return after the release (approx).

        Uses the storage effective date (release_date by default) and measures close-to-close return
        from release day to N trading days after.
        """
        try:
            key = "release_date" if self.use_storage_release_date else "date"
            events = storage_df[[key, "weekly_change_bcf"]].dropna().sort_values(key).copy()
            events[key] = pd.to_datetime(events[key])

            stock = stock_df.copy()
            if "Close" in stock.columns:
                close = stock["Close"]
            else:
                close = stock_df["Close"]

            rows = []
            for _, r in events.iterrows():
                d0 = r[key]
                if d0 not in close.index:
                    # pick next available trading day after d0
                    idx = close.index.searchsorted(d0)
                    if idx >= len(close.index):
                        continue
                    d0 = close.index[idx]
                # compute return over window_days trading sessions
                idx0 = close.index.get_loc(d0)
                idx1 = idx0 + window_days
                if idx1 >= len(close.index):
                    continue
                ret = close.iloc[idx1] / close.iloc[idx0] - 1
                rows.append({"event_date": d0, "weekly_change_bcf": r["weekly_change_bcf"], "return": ret})

            ev = pd.DataFrame(rows)
            if ev.empty:
                logger.info("No event windows available for plotting weekly change vs returns.")
                return

            plt.figure(figsize=(10, 6))
            plt.scatter(ev["weekly_change_bcf"], ev["return"], alpha=0.6)
            plt.xlabel("Weekly Storage Change (Bcf)")
            plt.ylabel(f"UNG Close Return over {window_days} trading day(s)")
            plt.title("UNG Return vs EIA Weekly Storage Change (Event Window)")

            plt.tight_layout()
            plt.savefig("UNG_Return_vs_WeeklyStorageChange_Event.png", dpi=300)
            plt.close()
            logger.info("Saved plot: UNG_Return_vs_WeeklyStorageChange_Event.png")
        except Exception as e:
            logger.error(f"Error plotting weekly change vs returns event window: {e}")

    # Backward-compatible alias
    def run_analysis(self):
        return self.run_analysis_temperature()


def main():
    """Main function to run analyses.

    By default, runs temperature lag analysis. Set RUN_STORAGE=True to also run storage analysis.
    """
    analyzer = UNGTemperatureAnalyzer(
        stock_file="ung_5year.csv",
        temp_file="tmp_2023-2024.csv",
        storage_file="eia_natural_gas_storage_4yr_complete.csv",
        use_storage_release_date=True
    )

    # Temperature lag analysis (same behavior as prior run_analysis)
    temp_results = analyzer.run_analysis_temperature()

    # Optional storage analysis
    RUN_STORAGE = True
    storage_results = None
    if RUN_STORAGE:
        storage_results = analyzer.run_analysis_storage()

    return temp_results, storage_results

if __name__ == "__main__":
    temp_results, storage_results = main()
