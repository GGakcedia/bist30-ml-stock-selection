import numpy as np
import pandas as pd

def equity_curve(daily_ret: pd.Series, start: float = 1.0) -> pd.Series:
    return start * (1.0 + daily_ret.fillna(0.0)).cumprod()

def sharpe(daily_ret: pd.Series, annualization: int = 252) -> float:
    r = daily_ret.dropna()
    if len(r) < 20:
        return float("nan")
    vol = r.std(ddof=0)
    if vol == 0:
        return float("nan")
    return float(np.sqrt(annualization) * r.mean() / vol)

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())