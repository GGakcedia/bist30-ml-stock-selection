"""
Downloads and caches historical data for the defined universe.
"""
import os
import pandas as pd
import yfinance as yf
from src.universe import BIST30_TICKERS

def download_and_cache_data(raw_data_dir: str, start_date: str = "2015-01-01") -> pd.DataFrame:
    """
    Downloads data for all BIST30 tickers and caches it locally to prevent
    redundant network calls and ensure reproducibility.
    """
    os.makedirs(raw_data_dir, exist_ok=True)
    cache_path = os.path.join(raw_data_dir, "bist30_raw.csv")

    if os.path.exists(cache_path):
        print(f"Loading data from cache: {cache_path}")
        df = pd.read_csv(cache_path, header=[0, 1], index_col=0, parse_dates=True)
        return df

    print(f"Downloading data for {len(BIST30_TICKERS)} tickers from {start_date}...")
    
    from curl_cffi import requests as requests_cffi
    session = requests_cffi.Session(impersonate="chrome", verify=False)

    df = yf.download(BIST30_TICKERS, start=start_date, group_by="ticker", auto_adjust=False, session=session)
    
    print(f"Saving data to {cache_path}")
    df.to_csv(cache_path)
    return df

def get_processed_dataframe(raw_data_dir: str, start_date: str = "2015-01-01") -> pd.DataFrame:
    """
    Loads data, flattens the multi-index columns, and formats it as a long-format panel.
    """
    df = download_and_cache_data(raw_data_dir, start_date)
    
    # yfinance returns a MultiIndex column DataFrame (Ticker, Price Type)
    # Target format should be a multi-index (date, ticker) with columns (open, high, low, close, adj_close, volume)
    
    # Stack level 0 (Ticker) into index to form (Date, Ticker)
    stacked = df.stack(level=0, future_stack=True)
    stacked.index.names = ["date", "ticker"]
    
    # Clean up column names to be lowercase
    stacked.columns = [c.lower().replace(" ", "_") for c in stacked.columns]
    
    # Drop rows where everything is NaN
    stacked = stacked.dropna(how="all").copy()
    
    # --- ML ROBUSTNESS IMPROVEMENTS ---
    
    # 1. Missing Data Handling: Forward-fill missing intermediate days per ticker (up to 5 days) 
    # Must group by ticker to prevent cross-sectional leakage.
    stacked = stacked.groupby(level="ticker").ffill(limit=5)
    
    # 2. Corporate Action Adjustment (Splits & Dividends)
    # yfinance provides unadjusted O,H,L,C. A 10:1 split causes a 90% artificial drop in Close.
    # Using 'adj_close / close' retroactively adjusts historical bars.
    if "adj_close" in stacked.columns and "close" in stacked.columns:
        adj_factor = stacked["adj_close"] / stacked["close"]
        
        for col in ["open", "high", "low", "close"]:
            if col in stacked.columns:
                stacked[col] = stacked[col] * adj_factor
                
        # Volume must be adjusted inversely to represent equivalent proportional liquidity
        if "volume" in stacked.columns:
            stacked["volume"] = stacked["volume"] / adj_factor
            
    # 3. Handle zero-volume edge cases (e.g. trading halts)
    if "volume" in stacked.columns:
        stacked["volume"] = stacked["volume"].replace(0, pd.NA)
        stacked["volume"] = stacked.groupby(level="ticker")["volume"].ffill(limit=5)
        
    # Drop any rows where we still don't have a valid close price
    stacked = stacked.dropna(subset=["close"])
    
    return stacked

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    df = get_processed_dataframe(raw_dir)
    print("Data loaded successfully.")
    print(df.head())
    print(df.info())
