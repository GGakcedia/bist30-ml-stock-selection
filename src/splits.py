"""
Walk-forward Time Series Split for Panel Data.
Ensures we split training/testing purely by date, keeping all tickers for a given date 
in the same split to avoid data leakage.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

class PanelTimeSeriesSplit:
    """
    Custom TimeSeriesSplit for Panel Data (e.g., MultiIndex of Date, Ticker).
    Splits are performed on the unique dates, and the resulting train/test 
    indices are mapped back to the original dataframe rows.
    """
    def __init__(self, n_splits=5, test_size=None, gap=0):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.tscv = TimeSeriesSplit(
            n_splits=n_splits, 
            test_size=test_size, 
            gap=gap
        )

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        """
        if not isinstance(X.index, pd.MultiIndex) or "date" not in X.index.names:
            raise ValueError("X must be a pandas DataFrame with a MultiIndex containing 'date'.")
        
        # Extract unique dates and sort them
        dates = pd.Series(X.index.get_level_values("date").unique().sort_values())
        
        for train_date_idx, test_date_idx in self.tscv.split(dates):
            train_dates = dates.iloc[train_date_idx]
            test_dates = dates.iloc[test_date_idx]
            
            # Get integer indices of rows matching these dates
            train_idx = np.where(X.index.get_level_values("date").isin(train_dates))[0]
            test_idx = np.where(X.index.get_level_values("date").isin(test_dates))[0]
            
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

if __name__ == "__main__":
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    tickers = ["AAPL", "MSFT"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    df = pd.DataFrame(np.random.randn(len(idx), 2), index=idx, columns=["f1", "f2"])
    
    splitter = PanelTimeSeriesSplit(n_splits=2, gap=1)
    for i, (tr_idx, te_idx) in enumerate(splitter.split(df)):
        tr_dates = df.iloc[tr_idx].index.get_level_values("date").unique()
        te_dates = df.iloc[te_idx].index.get_level_values("date").unique()
        print(f"Split {i}:")
        print(f"  Train: {tr_dates.min().date()} to {tr_dates.max().date()}")
        print(f"  Test:  {te_dates.min().date()} to {te_dates.max().date()}")
        assert tr_dates.max() < te_dates.min()
