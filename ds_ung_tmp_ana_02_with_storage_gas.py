import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_market_calendars import get_calendar
import numpy as np
import os
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UNGTemperatureAnalyzer:
    """Analyzer for price and temperature relationships (UNG or natural gas)."""
    
    def __init__(
        self,
        stock_file,
        temp_file,
        storage_file=None,
        use_storage_release_date=True,
        gas_file=None,
        price_label="UNG",
        output_prefix="ung",
        use_trading_calendar=True
    ):
        self.stock_file = stock_file
        self.temp_file = temp_file
        self.storage_file = storage_file
        self.use_storage_release_date = use_storage_release_date
        self.gas_file = gas_file
        self.price_label = price_label
        self.output_prefix = output_prefix
        self.use_trading_calendar = use_trading_calendar
        self.storage_data = None
        self.stock_data = None
        self.temp_data = None
        self.merged_data = None
        
        # Configuration
        self.config = {
            'analysis_period': {'start_day': 330, 'end_day': 420},
            'temperature_lags': [1, 3, 5, 7],
            'rolling_windows': [3, 5, 7, 30],
            'storage_lags_weeks': [1, 2, 4],
            'output_dir': 'output',
            'output_file': f'{self.output_prefix}_Temperature_Analysis_Results.csv'
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

    def load_gas_data(self):
        """Load and preprocess natural gas price data."""
        if not self.gas_file:
            raise ValueError("gas_file not provided")

        try:
            logger.info(f"Loading gas data from {self.gas_file}")
            df = pd.read_csv(self.gas_file, encoding="utf-8-sig")
            if df.empty:
                raise ValueError("Gas data file is empty")

            df = df.rename(columns={'Exchange Date': 'Date'})
            if 'Date' not in df.columns:
                raise ValueError("Gas data missing required column: Exchange Date")

            df['Date'] = pd.to_datetime(df['Date'], format="%d-%b-%Y")

            # Clean numeric columns (remove commas, percent signs, and plus symbols)
            numeric_cols = ['Close', 'Open', 'Low', 'High', 'Volume', 'OI', 'Bid', 'Ask', 'Net', '%Chg']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(',', '', regex=False)
                        .str.replace('%', '', regex=False)
                        .str.replace('+', '', regex=False)
                    )
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df['Year'] = df['Date'].dt.year
            df['Day_of_Year'] = df['Date'].dt.dayofyear
            df['Month'] = df['Date'].dt.strftime('%b')
            df['Month_Day'] = df['Date'].dt.strftime('%m-%d')

            start_date = pd.Timestamp("2023-01-01")
            df['Day_Count'] = (df['Date'] - start_date).dt.days + 1

            df = df.sort_values('Date').drop_duplicates(subset=['Date']).reset_index(drop=True)
            logger.info(f"Gas data loaded: {len(df)} rows, {df['Year'].nunique()} years")
            return df

        except Exception as e:
            logger.error(f"Error loading gas data: {e}")
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
            temp_files = self.temp_file if isinstance(self.temp_file, (list, tuple)) else [self.temp_file]
            logger.info(f"Loading temperature data from {temp_files}")
            frames = [pd.read_csv(path) for path in temp_files]
            df = pd.concat(frames, ignore_index=True)
            
            if df.empty:
                raise ValueError("Temperature data file is empty")
                
            # Preprocessing
            df['Date'] = pd.to_datetime(df['Date'], format="%Y%m%d")
            df['Year'] = df['Date'].dt.year
            df['Day_of_Year'] = df['Date'].dt.dayofyear
            
            # Create day count from 2023-01-01
            start_date = pd.Timestamp("2023-01-01")
            df['Day_Count'] = (df['Date'] - start_date).dt.days + 1
            
            # De-duplicate on Date in case of overlapping files
            df = df.sort_values('Date').drop_duplicates(subset=['Date']).reset_index(drop=True)
            logger.info(f"Temperature data loaded: {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error loading temperature data: {e}")
            raise

    def plot_temperature_series_comparison(self):
        """Plot two temperature series on a shared time axis with gaps outside each file's range."""
        temp_files = self.temp_file if isinstance(self.temp_file, (list, tuple)) else [self.temp_file]
        if len(temp_files) != 2:
            logger.warning("Temperature comparison plot expects exactly 2 files; skipping")
            return

        logger.info("Creating temperature series comparison plot")
        os.makedirs(self.config['output_dir'], exist_ok=True)
        frames = []
        labels = []
        for path in temp_files:
            df = pd.read_csv(path)
            if df.empty:
                raise ValueError(f"Temperature data file is empty: {path}")
            df['Date'] = pd.to_datetime(df['Date'], format="%Y%m%d")
            df = df.sort_values('Date').drop_duplicates(subset=['Date'])
            df['temperature_c'] = df['tm000'] - 273.15
            frames.append(df.set_index('Date')['temperature_c'])
            labels.append(path)

        start = min(series.index.min() for series in frames)
        end = max(series.index.max() for series in frames)
        full_index = pd.date_range(start=start, end=end, freq='D')
        aligned = [series.reindex(full_index) for series in frames]

        plt.figure(figsize=(12, 6))
        for series, label in zip(aligned, labels):
            plt.plot(series.index, series.values, label=label, linewidth=2, alpha=0.8)

        plt.title('Temperature Comparison (two periods)', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        output_path = os.path.join(self.config['output_dir'], f'{self.output_prefix}_temperature_series_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()

        logger.info("Temperature series comparison plot saved")
    
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
            
            if self.use_trading_calendar:
                # Get NYSE trading calendar
                nyse = get_calendar('NYSE')
                trading_days = nyse.valid_days(
                    start_date=temp_filtered.index.min(),
                    end_date=temp_filtered.index.max()
                ).tz_localize(None)
            else:
                trading_days = stock_filtered.index

            # Create full temperature timeline and align with trading days
            full_temp = temp_filtered.reindex(
                pd.date_range(
                    start=temp_filtered.index.min(),
                    end=temp_filtered.index.max(),
                    freq='D'
                )
            )
            full_temp.ffill(inplace=True)  # Forward fill missing values
            # Precompute 30-day rolling mean on daily data to avoid late starts on trading-day-only windows
            full_temp['temp_ma_30d'] = full_temp['tm000'].rolling(30).mean()
            aligned_temp = full_temp[full_temp.index.isin(trading_days)]
            
            # Merge datasets
            merged_df = stock_filtered.join(aligned_temp, how='left', rsuffix='_temp')
            
            # Handle any remaining missing values
            merged_df['tm000'] = merged_df['tm000'].interpolate(method='time')
            
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
            ma_col = f'temp_ma_{window}d'
            std_col = f'temp_std_{window}d'
            if ma_col not in df.columns:
                df[ma_col] = df['tm000'].rolling(window).mean()
            if std_col not in df.columns:
                df[std_col] = df['tm000'].rolling(window).std()
        
        # Temperature anomaly (deviation from 30-day average)
        df['temp_anomaly'] = df['tm000'] - df['temp_ma_30d']
        df['temp_anomaly_c'] = df['temperature_c'] - (df['temp_ma_30d'] - 273.15)

        # Lagged temperature anomaly (C)
        for lag in self.config['temperature_lags']:
            df[f'temp_anomaly_c_lag_{lag}d'] = df['temp_anomaly_c'].shift(lag)

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
        df['open_return_1d'] = df['Open'].pct_change()
        df['close_return_1d'] = df['Close'].pct_change()
        
        # Volume features
        df['volume_ma_5d'] = df['Volume'].rolling(5).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma_5d']
        
        # Price ranges
        df['daily_range'] = df['High'] - df['Low']
        df['range_pct'] = df['daily_range'] / df['Close']
        
        logger.info("Stock features created successfully")
        return df

    def create_storage_features(self, df):
        """Create lagged storage features (weekly cadence approximated in trading days)."""
        if 'storage_bcf' not in df.columns:
            logger.warning("Storage features not available; skipping")
            return df

        logger.info("Creating storage features")
        weeks_to_days = 5  # approximate one trading week
        for weeks in self.config['storage_lags_weeks']:
            shift_days = weeks * weeks_to_days
            df[f'storage_bcf_lag_{weeks}w'] = df['storage_bcf'].shift(shift_days)
            df[f'weekly_change_bcf_lag_{weeks}w'] = df['weekly_change_bcf'].shift(shift_days)
            if 'storage_change_pct' in df.columns:
                df[f'storage_change_pct_lag_{weeks}w'] = df['storage_change_pct'].shift(shift_days)

        logger.info("Storage features created successfully")
        return df
    
    def calculate_r_squared(self, df):
        """Calculate R-squared values for temperature features vs Open price"""
        logger.info("Calculating R-squared values")
        
        temperature_features = [
            'temperature_c',
            'temp_anomaly_c_lag_1d', 'temp_anomaly_c_lag_3d', 'temp_anomaly_c_lag_5d', 'temp_anomaly_c_lag_7d',
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
        plt.title(f'R-squared: Temperature Features vs {self.price_label} Open Price', 
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
        output_path = os.path.join(self.config['output_dir'], f'{self.output_prefix}_r_squared_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()
        
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
        plt.ylabel(f'{self.price_label} Open Price ($)', fontsize=12, fontweight='bold')
        plt.title(f'{self.price_label} Open Price vs {best_feature}\nR-squared = {r_squared_value:.3f}', 
                 fontsize=14, fontweight='bold')
        
        # Add equation to plot
        equation = f'y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}'
        plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', 
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(
            self.config['output_dir'],
            f'{self.output_prefix}_best_r_squared_relationship_{best_feature}.png'
        )
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()
        
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
        
        plt.title(f'{self.price_label} Open Price vs Temperature (5-day Lag)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save individual plot
        output_path = os.path.join(self.config['output_dir'], f'{self.output_prefix}_open_price_vs_temperature.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()
        
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
        output_path = os.path.join(self.config['output_dir'], f'{self.output_prefix}_correlation_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()
        
        logger.info("Correlation heatmap saved")

    def plot_price_vs_temp_anomaly_lag_correlation(self, df):
        """Plot correlation matrix: prices vs lagged temperature anomaly (C)."""
        logger.info("Creating price vs temp_anomaly_c lag correlation heatmap")

        lags = self.config['temperature_lags']
        for lag in lags:
            df[f'temp_anomaly_c_lag_{lag}d'] = df['temp_anomaly_c'].shift(lag)

        price_cols = ['Open', 'Close', 'High', 'Low']
        lag_cols = [f'temp_anomaly_c_lag_{lag}d' for lag in lags]

        corr_data = df[price_cols + lag_cols].corr()
        price_temp_corr = corr_data.loc[price_cols, lag_cols]

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            price_temp_corr,
            annot=True,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        plt.title('Price vs Lagged Temperature Anomaly (C)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        output_path = os.path.join(self.config['output_dir'], f'{self.output_prefix}_price_vs_temp_anomaly_lag_correlation.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()

        logger.info("Price vs temp_anomaly_c lag correlation heatmap saved")

    def plot_open_return_vs_temp_anomaly_lag_correlation(self, df):
        """Plot correlation heatmap: Open return vs lagged temperature anomaly (C)."""
        logger.info("Creating Open return vs temp_anomaly_c lag correlation heatmap")

        lags = self.config['temperature_lags']
        for lag in lags:
            df[f'temp_anomaly_c_lag_{lag}d'] = df['temp_anomaly_c'].shift(lag)

        lag_cols = [f'temp_anomaly_c_lag_{lag}d' for lag in lags]
        if 'open_return_1d' not in df.columns:
            logger.warning("open_return_1d missing; skipping Open return vs temp anomaly lag heatmap")
            return

        corr_data = df[['open_return_1d'] + lag_cols].corr()
        open_temp_corr = corr_data.loc[['open_return_1d'], lag_cols]

        plt.figure(figsize=(8, 3))
        sns.heatmap(
            open_temp_corr,
            annot=True,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        plt.title('Open Return vs Lagged Temperature Anomaly (C)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        output_path = os.path.join(
            self.config['output_dir'],
            f'{self.output_prefix}_open_return_vs_temp_anomaly_lag_correlation.png'
        )
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()

        logger.info("Open return vs temp_anomaly_c lag correlation heatmap saved")
    
    def plot_temperature_anomaly_vs_price_changes(self, df):
        """Plot 3: Temperature anomaly vs price changes scatter plot"""
        logger.info("Creating temperature anomaly vs price changes plot")

        if 'Volume' not in df.columns:
            logger.warning("Volume column missing; skipping temperature anomaly vs price changes plot")
            return

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
        output_path = os.path.join(self.config['output_dir'], f'{self.output_prefix}_temperature_anomaly_vs_price_changes.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()
        
        logger.info(f"Temperature anomaly vs price changes plot saved with {len(clean_data)} data points")
    
    def plot_volume_vs_temperature_volatility(self, df):
        """Plot 4: Volume vs temperature volatility"""
        logger.info("Creating volume vs temperature volatility plot")

        if 'Volume' not in df.columns:
            logger.warning("Volume column missing; skipping volume vs temperature volatility plot")
            return

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
        output_path = os.path.join(self.config['output_dir'], f'{self.output_prefix}_volume_vs_temperature_volatility.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()
        
        logger.info("Volume vs temperature volatility plot saved")

    def plot_open_price_vs_temperature_volatility(self, df):
        """Plot gas open price vs temperature volatility (5d)."""
        logger.info("Creating open price vs temperature volatility plot")

        clean_data = df[['Open', 'temp_volatility_5d']].dropna()
        if len(clean_data) == 0:
            logger.warning("No data available for open price vs temperature volatility plot")
            return

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color1 = 'tab:blue'
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Open Price ($)', color=color1, fontsize=12, fontweight='bold')
        ax1.plot(
            clean_data.index,
            clean_data['Open'],
            label='Open Price',
            color=color1,
            linewidth=2,
            alpha=0.8
        )
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Temperature Volatility (5-day STD)', color=color2, fontsize=12, fontweight='bold')
        ax2.plot(
            clean_data.index,
            clean_data['temp_volatility_5d'],
            label='Temp Volatility (5d)',
            color=color2,
            linestyle='--',
            linewidth=2,
            alpha=0.8
        )
        ax2.tick_params(axis='y', labelcolor=color2)

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.title('Gas Open Price vs Temperature Volatility (5d)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()

        output_path = os.path.join(self.config['output_dir'], f'{self.output_prefix}_open_price_vs_temp_volatility.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()

        logger.info("Open price vs temperature volatility plot saved")

    def plot_open_return_vs_temperature_volatility(self, df):
        """Plot gas open return vs temperature volatility (5d)."""
        logger.info("Creating open return vs temperature volatility plot")

        if 'open_return_1d' not in df.columns:
            logger.warning("Open return missing; skipping open return vs temperature volatility plot")
            return

        clean_data = df[['open_return_1d', 'temp_volatility_5d']].dropna()
        if len(clean_data) == 0:
            logger.warning("No data available for open return vs temperature volatility plot")
            return

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color1 = 'tab:blue'
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Open Return (1d)', color=color1, fontsize=12, fontweight='bold')
        ax1.plot(
            clean_data.index,
            clean_data['open_return_1d'],
            label='Open Return (1d)',
            color=color1,
            linewidth=2,
            alpha=0.8
        )
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Temperature Volatility (5-day STD)', color=color2, fontsize=12, fontweight='bold')
        ax2.plot(
            clean_data.index,
            clean_data['temp_volatility_5d'],
            label='Temp Volatility (5d)',
            color=color2,
            linestyle='--',
            linewidth=2,
            alpha=0.8
        )
        ax2.tick_params(axis='y', labelcolor=color2)

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.title('Gas Open Return vs Temperature Volatility (5d)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()

        output_path = os.path.join(self.config['output_dir'], f'{self.output_prefix}_open_return_vs_temp_volatility.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()

        logger.info("Open return vs temperature volatility plot saved")
    def plot_open_price_vs_temp_anomaly_lag_5d(self, df):
        """Plot gas open price vs temperature anomaly (C) lagged 5 days."""
        logger.info("Creating open price vs temp_anomaly_c_lag_5d plot")

        if 'temp_anomaly_c_lag_5d' not in df.columns:
            logger.warning("temp_anomaly_c_lag_5d missing; skipping open price vs temp anomaly lag plot")
            return

        clean_data = df[['Open', 'temp_anomaly_c_lag_5d']].dropna()
        if len(clean_data) == 0:
            logger.warning("No data available for open price vs temp anomaly lag plot")
            return

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color1 = 'tab:blue'
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Open Price ($)', color=color1, fontsize=12, fontweight='bold')
        ax1.plot(
            clean_data.index,
            clean_data['Open'],
            label='Open Price',
            color=color1,
            linewidth=2,
            alpha=0.8
        )
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Temp Anomaly (C) Lag 5d', color=color2, fontsize=12, fontweight='bold')
        ax2.plot(
            clean_data.index,
            clean_data['temp_anomaly_c_lag_5d'],
            label='Temp Anomaly (C) Lag 5d',
            color=color2,
            linestyle='--',
            linewidth=2,
            alpha=0.8
        )
        ax2.tick_params(axis='y', labelcolor=color2)

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.title('Gas Open Price vs Temp Anomaly (C) Lag 5d', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()

        output_path = os.path.join(self.config['output_dir'], f'{self.output_prefix}_open_price_vs_temp_anomaly_c_lag_5d.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()

        logger.info("Open price vs temp_anomaly_c_lag_5d plot saved")
    
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

    def analyze_storage_correlations(self, df):
        """Correlate UNG returns vs storage levels and weekly changes."""
        logger.info("Analyzing storage correlations")

        if 'storage_bcf' not in df.columns:
            logger.warning("Storage data not present; skipping storage correlation analysis")
            return None, None

        return_cols = ['open_return_1d', 'close_return_1d']
        storage_cols = [
            'storage_bcf', 'weekly_change_bcf', 'storage_change_pct'
        ]
        lag_cols = []
        for weeks in self.config['storage_lags_weeks']:
            lag_cols.extend([
                f'storage_bcf_lag_{weeks}w',
                f'weekly_change_bcf_lag_{weeks}w',
                f'storage_change_pct_lag_{weeks}w'
            ])

        analysis_cols = [c for c in return_cols + storage_cols + lag_cols if c in df.columns]
        corr_matrix = df[analysis_cols].corr()

        significant = []
        for rcol in return_cols:
            if rcol not in df.columns:
                continue
            for scol in storage_cols + lag_cols:
                if scol not in df.columns:
                    continue
                valid = df[[rcol, scol]].dropna()
                if len(valid) > 2:
                    corr, p_value = pearsonr(valid[rcol], valid[scol])
                    significant.append({
                        'return_feature': rcol,
                        'storage_feature': scol,
                        'correlation': corr,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })

        significant_df = pd.DataFrame(significant)
        logger.info("Storage correlation analysis completed")
        return corr_matrix, significant_df

    def plot_storage_correlation_heatmap(self, df):
        """Plot correlation heatmap: returns vs storage features and lags."""
        if 'storage_bcf' not in df.columns:
            logger.warning("Storage data not present; skipping storage heatmap")
            return

        return_cols = [c for c in ['open_return_1d', 'close_return_1d'] if c in df.columns]
        storage_cols = [
            'storage_bcf', 'weekly_change_bcf', 'storage_change_pct'
        ]
        lag_cols = []
        for weeks in self.config['storage_lags_weeks']:
            lag_cols.extend([
                f'storage_bcf_lag_{weeks}w',
                f'weekly_change_bcf_lag_{weeks}w',
                f'storage_change_pct_lag_{weeks}w'
            ])

        cols = [c for c in return_cols + storage_cols + lag_cols if c in df.columns]
        corr_data = df[cols].corr()
        heatmap_data = corr_data.loc[return_cols, [c for c in storage_cols + lag_cols if c in corr_data.columns]]

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            heatmap_data,
            annot=True,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=False,
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        plt.title(f'{self.price_label} Returns vs Storage Levels/Changes', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        output_path = os.path.join(self.config['output_dir'], f'{self.output_prefix}_returns_vs_storage_correlation.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()

        logger.info("Storage correlation heatmap saved")

    def plot_weekly_change_vs_open_price(self, df):
        """Plot weekly storage change (BCF) against gas open price."""
        if 'weekly_change_bcf' not in df.columns:
            logger.warning("Weekly storage change missing; skipping weekly change vs open price plot")
            return

        clean_data = df[['Open', 'weekly_change_bcf']].dropna()
        if len(clean_data) == 0:
            logger.warning("No data available for weekly change vs open price plot")
            return

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color1 = 'tab:blue'
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Open Price ($)', color=color1, fontsize=12, fontweight='bold')
        ax1.plot(
            clean_data.index,
            clean_data['Open'],
            label='Open Price',
            color=color1,
            linewidth=2,
            alpha=0.8
        )
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Weekly Change (BCF)', color=color2, fontsize=12, fontweight='bold')
        ax2.plot(
            clean_data.index,
            clean_data['weekly_change_bcf'],
            label='Weekly Change (BCF)',
            color=color2,
            linestyle='--',
            linewidth=2,
            alpha=0.8
        )
        ax2.tick_params(axis='y', labelcolor=color2)

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.title('Weekly Storage Change vs Gas Open Price', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()

        output_path = os.path.join(self.config['output_dir'], f'{self.output_prefix}_weekly_change_vs_open_price.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()

        logger.info("Weekly change vs open price plot saved")

    def plot_weekly_change_vs_open_return(self, df):
        """Plot weekly storage change (BCF) against gas open return."""
        if 'weekly_change_bcf' not in df.columns:
            logger.warning("Weekly storage change missing; skipping weekly change vs open return plot")
            return
        if 'open_return_1d' not in df.columns:
            logger.warning("Open return missing; skipping weekly change vs open return plot")
            return

        clean_data = df[['open_return_1d', 'weekly_change_bcf']].dropna()
        if len(clean_data) == 0:
            logger.warning("No data available for weekly change vs open return plot")
            return

        fig, ax1 = plt.subplots(figsize=(12, 6))

        color1 = 'tab:blue'
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Open Return (1d)', color=color1, fontsize=12, fontweight='bold')
        ax1.plot(
            clean_data.index,
            clean_data['open_return_1d'],
            label='Open Return (1d)',
            color=color1,
            linewidth=2,
            alpha=0.8
        )
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Weekly Change (BCF)', color=color2, fontsize=12, fontweight='bold')
        ax2.plot(
            clean_data.index,
            clean_data['weekly_change_bcf'],
            label='Weekly Change (BCF)',
            color=color2,
            linestyle='--',
            linewidth=2,
            alpha=0.8
        )
        ax2.tick_params(axis='y', labelcolor=color2)

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.title('Weekly Storage Change vs Gas Open Return', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()

        output_path = os.path.join(self.config['output_dir'], f'{self.output_prefix}_weekly_change_vs_open_return.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #plt.show()

        logger.info("Weekly change vs open return plot saved")

    def run_multivariate_regression(self, df):
        """Run multivariate regression on returns with time-series validation.

        This reduces spurious correlations from non-stationary price levels by modeling
        return dynamics against lagged temperature and storage features.
        """
        logger.info("Running multivariate regression (returns)")

        base_features = [
            'temp_c_lag_1d', 'temp_c_lag_3d', 'temp_c_lag_5d', 'temp_c_lag_7d',
            'temp_anomaly_c', 'temp_volatility_5d'
        ]
        storage_features = ['storage_bcf', 'weekly_change_bcf', 'storage_change_pct']
        feature_cols = [c for c in base_features if c in df.columns]
        if self.storage_file:
            feature_cols.extend([c for c in storage_features if c in df.columns])

        target_col = 'open_return_1d'
        data = df[feature_cols + [target_col]].dropna()
        if len(data) < 30:
            logger.warning("Not enough data for multivariate regression")
            return None

        X = data[feature_cols].values
        y = data[target_col].values

        # Time-series cross-validation (walk-forward)
        n_splits = min(5, max(2, len(data) // 20))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oos_scores = []
        for train_idx, test_idx in tscv.split(X):
            model = LinearRegression()
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[test_idx])
            oos_scores.append(r2_score(y[test_idx], y_pred))

        # Fit on full data for coefficients
        final_model = LinearRegression()
        final_model.fit(X, y)
        coef_df = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': final_model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)

        results = {
            'in_sample_r2': final_model.score(X, y),
            'oos_r2_mean': float(np.mean(oos_scores)),
            'oos_r2_std': float(np.std(oos_scores)),
            'n_samples': len(data),
            'feature_count': len(feature_cols),
            'coefficients': coef_df
        }

        logger.info("Multivariate regression completed")
        return results
    
    def generate_report(self, df, corr_matrix, significant_correlations, r_squared_df, multivar_results, storage_significant):
        """Generate analysis report with R-squared results"""
        logger.info("Generating analysis report")
        
        print("=" * 70)
        print(f"{self.price_label.upper()} PRICE & TEMPERATURE ANALYSIS REPORT")
        print("=" * 70)
        
        print(f"\nDATA OVERVIEW:")
        print(f"Analysis period: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        print(f"Total trading days: {len(df)}")
        print(f"Average Open Price: ${df['Open'].mean():.2f}")
        print(f"Average Temperature: {df['temperature_c'].mean():.1f}°C")
        
        print(f"\nR-SQUARED ANALYSIS (Temperature vs {self.price_label} Open Price):")
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

        print(f"\nMULTIVARIATE RETURNS MODEL:")
        print("-" * 50)
        if multivar_results:
            print(f"Samples: {multivar_results['n_samples']}, Features: {multivar_results['feature_count']}")
            print(f"In-sample R²: {multivar_results['in_sample_r2']:.3f}")
            print(f"OOS R² (mean ± std): {multivar_results['oos_r2_mean']:.3f} ± {multivar_results['oos_r2_std']:.3f}")
            print("Top coefficients (by magnitude):")
            for _, row in multivar_results['coefficients'].head(5).iterrows():
                print(f"  {row['feature']}: {row['coefficient']:.4f}")
        else:
            print("Not enough data to fit a multivariate returns model")

        print(f"\nSTORAGE vs RETURNS:")
        print("-" * 50)
        if storage_significant is not None and len(storage_significant) > 0:
            storage_significant = storage_significant.sort_values('correlation', key=abs, ascending=False)
            for _, row in storage_significant.head(10).iterrows():
                significance = "***" if row['significant'] else ""
                print(f"{row['return_feature']} vs {row['storage_feature']}: "
                      f"{row['correlation']:.3f} (p={row['p_value']:.3f}){significance}")
        else:
            print("No significant storage/return correlations found")

        print(f"\nKEY INSIGHTS:")
        if len(r_squared_df) > 0:
            best_feature = r_squared_df.iloc[0]
            print(f"• Temperature explains {best_feature['r_squared']*100:.1f}% of {self.price_label} price variance")
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
    
    def save_results(self, df, r_squared_df, multivar_results, storage_significant):
        """Save analysis results to file"""
        try:
            output_dir = self.config['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            # Save main data
            output_file = os.path.join(output_dir, self.config['output_file'])
            df.to_csv(output_file)
            logger.info(f"Analysis results saved to {output_file}")
            
            # Save R-squared results
            r_squared_file = os.path.join(output_dir, f'{self.output_prefix}_r_squared_results.csv')
            r_squared_df.to_csv(r_squared_file, index=False)
            logger.info(f"R-squared results saved to {r_squared_file}")

            if multivar_results:
                multivar_file = os.path.join(output_dir, f'{self.output_prefix}_multivariate_return_coefficients.csv')
                multivar_results['coefficients'].to_csv(multivar_file, index=False)
                logger.info(f"Multivariate coefficients saved to {multivar_file}")

            if storage_significant is not None:
                storage_file = os.path.join(output_dir, f'{self.output_prefix}_storage_return_correlations.csv')
                storage_significant.to_csv(storage_file, index=False)
                logger.info(f"Storage/return correlations saved to {storage_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def run_analysis(self):
        """Main method to run complete analysis"""
        try:
            logger.info(f"Starting {self.price_label} Temperature Analysis")
            
            # Load data
            if self.gas_file:
                self.stock_data = self.load_gas_data()
            else:
                self.stock_data = self.load_stock_data()
            self.temp_data = self.load_temperature_data()
            self.plot_temperature_series_comparison()
            
            # Merge datasets
            self.merged_data = self.merge_datasets()

            # Load and merge weekly EIA storage data (optional)
            if self.storage_file:
                self.storage_data = self.load_storage_data()
                self.merged_data = self.merge_storage_to_trading_days(self.merged_data, self.storage_data)
            
            # Create features
            self.merged_data = self.create_temperature_features(self.merged_data)
            self.merged_data = self.create_stock_features(self.merged_data)
            self.merged_data = self.create_storage_features(self.merged_data)

            # Ensure output directory exists for plots and CSVs
            os.makedirs(self.config['output_dir'], exist_ok=True)

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
            self.plot_price_vs_temp_anomaly_lag_correlation(self.merged_data)
            self.plot_open_return_vs_temp_anomaly_lag_correlation(self.merged_data)
            self.plot_storage_correlation_heatmap(self.merged_data)
            self.plot_weekly_change_vs_open_price(self.merged_data)
            self.plot_weekly_change_vs_open_return(self.merged_data)
            self.plot_temperature_anomaly_vs_price_changes(self.merged_data)
            self.plot_volume_vs_temperature_volatility(self.merged_data)
            self.plot_open_price_vs_temperature_volatility(self.merged_data)
            self.plot_open_return_vs_temperature_volatility(self.merged_data)
            self.plot_open_price_vs_temp_anomaly_lag_5d(self.merged_data)
            
            # Generate report
            multivar_results = self.run_multivariate_regression(self.merged_data)
            storage_corr_matrix, storage_significant = self.analyze_storage_correlations(self.merged_data)
            self.generate_report(self.merged_data, corr_matrix, significant_correlations, r_squared_df, multivar_results, storage_significant)
            
            # Save results
            self.save_results(self.merged_data, r_squared_df, multivar_results, storage_significant)
            
            logger.info("Analysis completed successfully")
            return self.merged_data, r_squared_df
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

def main():
    """Main function to run the analysis"""
    analyzer = UNGTemperatureAnalyzer(
        stock_file="ung_5year.csv",
        temp_file=["tmp_2023-2024.csv", "tmp_202412-202502.csv"],
        storage_file="eia_natural_gas_storage_4yr_complete.csv",
        use_storage_release_date=True,
        gas_file="gas_price_History_20260125_1040.csv",
        price_label="Natural Gas",
        output_prefix="gas",
        use_trading_calendar=False
    )
    
    results, r_squared_results = analyzer.run_analysis()
    return results, r_squared_results

if __name__ == "__main__":
    results, r_squared_results = main()
