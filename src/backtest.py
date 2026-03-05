from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation import equity_curve, sharpe, max_drawdown

ROOT_RESULTS = Path("results")
METRICS_DIR = ROOT_RESULTS / "metrics"
BACKTEST_DIR = ROOT_RESULTS / "backtests"
FIG_DIR = ROOT_RESULTS / "figures"
for p in [METRICS_DIR, BACKTEST_DIR, FIG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

def backtest_topk(
    oof: pd.DataFrame,
    k: int = 5,
    cost_bps: float = 0.0,
) -> dict:
    """
    Daily Top-K long, equal-weight within selected names.
    Uses OOF pred_prob to rank tickers cross-sectionally each day.

    Transaction cost approximation:
    - compute daily turnover as fraction of names changed in Top-K set
    - apply cost_bps * turnover to daily return
    """
    df = oof.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["date", "pred_prob"], ascending=[True, False]).reset_index(drop=True)

    # pick top-k tickers per day
    topk = df.groupby("date").head(k).copy()

    # gross daily return: average next-day return of selected names
    daily = topk.groupby("date")["target_return"].mean().rename("gross_ret").to_frame()

    # turnover: how many names changed vs previous day
    # compute sets per day
    day_to_set = topk.groupby("date")["ticker"].apply(lambda x: set(x.tolist()))
    dates = day_to_set.index.to_list()

    turnovers = []
    prev_set = None
    for d in dates:
        cur = day_to_set.loc[d]
        if prev_set is None:
            t = 0.0
        else:
            # turnover fraction of changed names out of K
            changed = len(cur.symmetric_difference(prev_set))
            # symmetric_difference counts both directions; divide by 2 to count replacements
            t = (changed / 2) / float(k)
        turnovers.append(t)
        prev_set = cur

    daily["turnover"] = turnovers
    daily["cost"] = daily["turnover"] * (cost_bps / 10000.0)
    daily["net_ret"] = daily["gross_ret"] - daily["cost"]
    daily["equity"] = equity_curve(daily["net_ret"], start=1.0)

    stats = {
        "strategy": f"top{k}",
        "k": int(k),
        "cost_bps": float(cost_bps),
        "n_days": int(len(daily)),
        "cumulative_return": float(daily["equity"].iloc[-1] - 1.0),
        "sharpe": sharpe(daily["net_ret"]),
        "max_drawdown": max_drawdown(daily["equity"]),
        "avg_daily_ret": float(daily["net_ret"].mean()),
        "daily_vol": float(daily["net_ret"].std(ddof=0)),
        "avg_turnover": float(daily["turnover"].mean()),
    }

    return {"daily": daily.reset_index(), "stats": stats, "selected": topk}

def backtest_topk_weekly(oof, k=5, cost_bps=10):

    df = oof.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "pred_prob"], ascending=[True, False])

    df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)

    weekly_returns = []
    prev_set = None
    turnovers = []

    for week, group in df.groupby("week"):

        first_day = group["date"].min()

        day_df = group[group["date"] == first_day]

        topk = day_df.nlargest(k, "pred_prob")

        tickers = set(topk["ticker"])

        if prev_set is None:
            turnover = 0
        else:
            turnover = len(tickers.symmetric_difference(prev_set)) / (2 * k)

        prev_set = tickers
        turnovers.append(turnover)

        # apply for all days that week
        week_data = group[group["ticker"].isin(tickers)]

        daily_ret = week_data.groupby("date")["target_return"].mean()

        weekly_returns.extend(daily_ret.values)

    daily = pd.DataFrame({"ret": weekly_returns})

    daily["cost"] = np.mean(turnovers) * (cost_bps / 10000)

    daily["net_ret"] = daily["ret"] - daily["cost"]

    daily["equity"] = equity_curve(daily["net_ret"], start=1.0)

    stats = {
        "k": k,
        "cost_bps": cost_bps,
        "cumulative_return": daily["equity"].iloc[-1] - 1,
        "sharpe": sharpe(daily["net_ret"]),
        "max_drawdown": max_drawdown(daily["equity"]),
        "avg_turnover": np.mean(turnovers),
    }

    return {"daily": daily, "stats": stats}

def make_equal_weight_benchmark(oof: pd.DataFrame) -> pd.DataFrame:
    """
    Benchmark: Every day equal-weight long across all tickers.
    Uses target_return (next-day return) from the OOF file.
    """
    df = oof.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    daily = df.groupby("date")["target_return"].mean().rename("bench_ret").to_frame()
    daily["bench_equity"] = equity_curve(daily["bench_ret"], start=1.0)
    return daily.reset_index()

def plot_strategy_vs_benchmark(strategy_daily: pd.DataFrame, bench_daily: pd.DataFrame, out_path: Path) -> None:
    merged = pd.merge(strategy_daily, bench_daily, on="date", how="inner")

    plt.figure()
    plt.plot(merged["date"], merged["equity"], label="Strategy (Long/Cash)")
    plt.plot(merged["date"], merged["bench_equity"], label="Benchmark (Equal-Weight Long)")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.title("Strategy vs Benchmark")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    # Paths
    oof_path = METRICS_DIR / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"Missing: {oof_path}")

    oof = pd.read_csv(oof_path)

    required = {"ticker", "date", "target", "pred_prob", "pred", "target_return"}
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"Missing columns in oof_predictions.csv: {missing}")

    # Benchmark (always-long equal weight)
    bench = make_equal_weight_benchmark(oof)

    # --- COST SWEEP SETTINGS ---
    k_list = [5, 10]
    cost_list = [0.0, 10.0, 20.0, 30.0]

    rows = []
    best = None

    for k in k_list:
        for cost_bps in cost_list:
            res = backtest_topk(oof, k=k, cost_bps=cost_bps)
            s = res["stats"]
            rows.append(s)

            # pick best by Sharpe (robustness: ignore NaN)
            if best is None:
                best = res
            else:
                cur = np.nan_to_num(s["sharpe"], nan=-999)
                prev = np.nan_to_num(best["stats"]["sharpe"], nan=-999)
                if cur > prev:
                    best = res

    summary = pd.DataFrame(rows).sort_values(["k", "cost_bps"], ascending=[True, True])

    # Save sweep table
    sweep_path = BACKTEST_DIR / "topk_cost_sweep.csv"
    summary.to_csv(sweep_path, index=False)

    # Save best run artifacts
    best_daily_path = BACKTEST_DIR / "best_topk_daily.csv"
    best_stats_path = BACKTEST_DIR / "best_topk_stats.json"
    best_fig_path = FIG_DIR / "equity_curve_best_topk_vs_benchmark.png"

    best["daily"].to_csv(best_daily_path, index=False)
    best_stats_path.write_text(json.dumps(best["stats"], indent=2), encoding="utf-8")
    plot_strategy_vs_benchmark(best["daily"], bench, best_fig_path)

    print("\nSaved files:")
    print(f"- {sweep_path}")
    print(f"- {best_daily_path}")
    print(f"- {best_stats_path}")
    print(f"- {best_fig_path}")

    # Print a compact view
    print("\nTopK cost sweep (sorted by k, cost):")
    cols = ["strategy", "k", "cost_bps", "cumulative_return", "sharpe", "max_drawdown", "avg_turnover"]
    print(summary[cols].to_string(index=False))

    # Also print best config
    print("\nBest config by Sharpe:")
    print(best["stats"])

if __name__ == "__main__":
    main()