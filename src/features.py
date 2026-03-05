"""
Feature engineering module for BIST30 Direction model.
"""
import pandas as pd
import pandas_ta as ta
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes features for a stacked dataframe (Date, Ticker).
    Input `df` must have columns: ['open', 'high', 'low', 'close', 'adj_close', 'volume'].
    """
    ticker_dfs = []

    # Process features per ticker
    for ticker in df.index.get_level_values("ticker").unique():
        group = df.xs(ticker, level="ticker").copy()
        
        # CRITICAL FIX 1: Always sort chronologically before time-based shifts. 
        # If the index gets scrambled upstream, shift(-1) will pull random future 
        # or historical data into day T, causing massive look-ahead bias leakage.
        group = group.sort_index()
        
        # 1. Labels: We want to predict Next-Day Direction based on Adjusted Close returns
        # Label = 1 if Tomorrow's Return > 0 else 0
        group["target_return"] = group["adj_close"].pct_change().shift(-1)
        group["target_direction"] = (group["target_return"] > 0).astype(int)
        
        # 2. Price/Volume Features
        group["return_1d"] = group["adj_close"].pct_change()
        group["reversal_1d"] = -group["return_1d"]
        group["return_5d"] = group["adj_close"].pct_change(5)
        group["return_21d"] = group["adj_close"].pct_change(21)
        group["ret_vol_adj_21d"] = group["return_21d"] / group["volatility_21d"]
        
        if "volume" in group.columns:
            # Shift back so we only use today's volume compared to yesterday
            group["volume_change_1d"] = group["volume"].pct_change()
            group["volume_ma_10_ratio"] = group["volume"] / group["volume"].rolling(10).mean()
            group["volume_zscore"] = (
    group["volume"] - group["volume"].rolling(20).mean()
) / group["volume"].rolling(20).std()
        
        # 3. Technical Indicators (using pandas_ta)
        group["rsi_14"] = ta.rsi(group["adj_close"], length=14)
        
        # MACD
        macd = ta.macd(group["adj_close"])
        if macd is not None:
            # macd columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            group = pd.concat([group, macd.iloc[:, 0].rename("macd")], axis=1)
        
        # Volatility
        group["volatility_21d"] = group["return_1d"].rolling(21).std() * np.sqrt(252)
        
        # Moving Averages distance
        group["sma_50"] = ta.sma(group["adj_close"], length=50)
        group["sma_200"] = ta.sma(group["adj_close"], length=200)
        group["dist_sma_50"] = group["adj_close"] / group["sma_50"] - 1
        group["dist_sma_200"] = group["adj_close"] / group["sma_200"] - 1

        # Re-add ticker back to index
        group["ticker"] = ticker
        group = group.set_index("ticker", append=True)
        # Swap levels to be (Date, Ticker) again
        group = group.swaplevel(0, 1)

        ticker_dfs.append(group)
        
    features_df = pd.concat(ticker_dfs)
    
    # 4. Cross-Sectional Ranking Features (Market neutralizing)
    # Rank RSI and Returns daily across all available tickers
    date_groups = features_df.groupby(level="date")
    features_df["rsi_14_rank_cs"] = date_groups["rsi_14"].rank(pct=True)
    features_df["return_1d_rank_cs"] = date_groups["return_1d"].rank(pct=True)
    features_df["return_21d_rank_cs"] = date_groups["return_21d"].rank(pct=True)
    features_df["volatility_rank_cs"] = date_groups["volatility_21d"].rank(pct=True)
    features_df["dist_sma_50_rank_cs"] = date_groups["dist_sma_50"].rank(pct=True)
    features_df["volume_rank_cs"] = date_groups["volume_ma_10_ratio"].rank(pct=True)
    
    # Drop rows with mostly NaNs (due to MA 200 and shifted target)
    features_df = features_df.dropna(subset=["sma_200", "target_return"])
    
    return features_df

if __name__ == "__main__":
    from src.data_loader import get_processed_dataframe
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = get_processed_dataframe(os.path.join(base_dir, "data", "raw"))
    print("Computing features...")
    feat_df = build_features(df)
    print(feat_df.head(10))
    print("Feature columns:", feat_df.columns.tolist())
