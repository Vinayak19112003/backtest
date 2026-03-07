"""
Phase 5: Comprehensive strategy ranking and composite scoring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"
EXH_DIR = REPORTS_DIR / "exhaustive_filter_analysis"

INPUT_PATH = EXH_DIR / "10_comparative_matrix.csv"
RANKINGS_PATH = REPORTS_DIR / "strategy_rankings_all_metrics.csv"
COMPOSITE_PATH = REPORTS_DIR / "composite_scores.csv"
TOP5_PATH = REPORTS_DIR / "top_5_comparison_matrix.csv"


RANK_SPECS = {
    "rank_total_pnl": ("total_pnl", False),
    "rank_sharpe_ratio": ("sharpe_ratio", False),
    "rank_win_rate": ("win_rate", False),
    "rank_max_drawdown_pct": ("max_drawdown_pct", True),
    "rank_consistency_score": ("consistency_score", False),
    "rank_trades_per_day": ("trades_per_day", False),
}


COMPONENT_COLUMNS = [
    "strategy_name",
    "total_pnl",
    "sharpe_ratio",
    "win_rate",
    "max_drawdown_pct",
    "trades_per_day",
    "consistency_score",
]


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize to 0-100 scale."""
    if max_val == min_val:
        return 50.0
    return float(((value - min_val) / (max_val - min_val)) * 100)


def composite_score(row: pd.Series, all_strategies: pd.DataFrame) -> dict[str, float]:
    """
    Weighted scoring for best all-rounder Polymarket strategy.
    """
    pnl_score = normalize(
        row["total_pnl"],
        all_strategies["total_pnl"].min(),
        all_strategies["total_pnl"].max(),
    )

    sharpe_score = normalize(
        row["sharpe_ratio"],
        all_strategies["sharpe_ratio"].min(),
        all_strategies["sharpe_ratio"].max(),
    )

    wr_score = normalize(
        row["win_rate"],
        all_strategies["win_rate"].min(),
        all_strategies["win_rate"].max(),
    )

    dd_score = 100 - normalize(
        row["max_drawdown_pct"],
        all_strategies["max_drawdown_pct"].min(),
        all_strategies["max_drawdown_pct"].max(),
    )

    capacity_score = normalize(
        row["trades_per_day"],
        all_strategies["trades_per_day"].min(),
        all_strategies["trades_per_day"].max(),
    )

    consistency_score_norm = normalize(
        row["consistency_score"],
        all_strategies["consistency_score"].min(),
        all_strategies["consistency_score"].max(),
    )

    weights = {
        "pnl": 0.25,
        "sharpe": 0.25,
        "wr": 0.15,
        "dd": 0.20,
        "capacity": 0.10,
        "consistency": 0.05,
    }

    composite = (
        weights["pnl"] * pnl_score
        + weights["sharpe"] * sharpe_score
        + weights["wr"] * wr_score
        + weights["dd"] * dd_score
        + weights["capacity"] * capacity_score
        + weights["consistency"] * consistency_score_norm
    )

    return {
        "pnl_score": pnl_score,
        "sharpe_score": sharpe_score,
        "win_rate_score": wr_score,
        "drawdown_score": dd_score,
        "capacity_score": capacity_score,
        "consistency_score_norm": consistency_score_norm,
        "composite_score": composite,
    }


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in COMPONENT_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0 if col != "strategy_name" else "UNKNOWN"
    for col in COMPONENT_COLUMNS:
        if col == "strategy_name":
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df = _ensure_columns(df)

    rankings = df.copy()
    for rank_col, (metric_col, ascending) in RANK_SPECS.items():
        rankings[rank_col] = rankings[metric_col].rank(method="min", ascending=ascending).astype(int)

    rankings.to_csv(RANKINGS_PATH, index=False)

    components = []
    for _, row in df.iterrows():
        scores = composite_score(row, df)
        components.append(
            {
                "strategy_name": row["strategy_name"],
                **scores,
                "total_pnl": row["total_pnl"],
                "sharpe_ratio": row["sharpe_ratio"],
                "win_rate": row["win_rate"],
                "max_drawdown_pct": row["max_drawdown_pct"],
                "trades_per_day": row["trades_per_day"],
                "consistency_score": row["consistency_score"],
            }
        )

    composite_df = pd.DataFrame(components).sort_values("composite_score", ascending=False)
    composite_df["rank_composite"] = range(1, len(composite_df) + 1)
    composite_df.to_csv(COMPOSITE_PATH, index=False)

    top5 = composite_df.head(5).copy()
    top5 = top5[
        [
            "rank_composite",
            "strategy_name",
            "composite_score",
            "total_pnl",
            "sharpe_ratio",
            "win_rate",
            "max_drawdown_pct",
            "trades_per_day",
            "consistency_score",
        ]
    ]
    top5.to_csv(TOP5_PATH, index=False)

    print(f"Saved: {RANKINGS_PATH}")
    print(f"Saved: {COMPOSITE_PATH}")
    print(f"Saved: {TOP5_PATH}")


if __name__ == "__main__":
    main()
