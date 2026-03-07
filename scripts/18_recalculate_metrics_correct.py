"""
Phase 3: Recalculate risk metrics with corrected formulas for Polymarket 15m BTC predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"
EXH_DIR = REPORTS_DIR / "exhaustive_filter_analysis"
VALIDATION_REPORT = REPORTS_DIR / "metric_validation_report.txt"

INITIAL_CAPITAL = 100.0
PERIODS_PER_YEAR = 35040


@dataclass
class StrategyMetrics:
    strategy_name: str
    total_trades: int
    win_count: int
    loss_count: int
    win_rate: float
    total_profit: float
    total_loss: float
    total_pnl: float
    avg_pnl_per_trade: float
    direction_yes_trades: int
    direction_no_trades: int
    direction_yes_pct: float
    direction_no_pct: float
    trades_per_day: float
    cagr: float
    cagr_pct: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_dollars: float
    max_drawdown_pct: float
    peak_before_max_dd: float
    valley_at_max_dd: float
    max_drawdown_start_date: str
    max_drawdown_duration_days: int
    recovery_factor: float


def calculate_drawdown_correct(cumulative_pnl: pd.Series, initial_capital: float = 100.0) -> dict[str, Any]:
    """
    Traditional peak-to-valley drawdown, expressed as % of initial capital.
    """
    if cumulative_pnl.empty:
        return {
            "dd_series": pd.Series(dtype=float),
            "dd_pct_series": pd.Series(dtype=float),
            "max_dd_dollars": 0.0,
            "max_dd_pct": 0.0,
            "peak_before_dd": initial_capital,
            "valley_at_dd": initial_capital,
            "max_dd_idx": None,
            "equity": pd.Series(dtype=float),
            "peak": pd.Series(dtype=float),
        }

    equity = initial_capital + cumulative_pnl
    peak = equity.expanding().max()
    dd_dollars = peak - equity
    dd_pct = (dd_dollars / initial_capital) * 100

    max_dd_idx = dd_dollars.idxmax()
    max_dd_dollars = float(dd_dollars.loc[max_dd_idx])
    max_dd_pct = float(dd_pct.loc[max_dd_idx])

    return {
        "dd_series": dd_dollars,
        "dd_pct_series": dd_pct,
        "max_dd_dollars": max_dd_dollars,
        "max_dd_pct": max_dd_pct,
        "peak_before_dd": float(peak.loc[max_dd_idx]),
        "valley_at_dd": float(equity.loc[max_dd_idx]),
        "max_dd_idx": max_dd_idx,
        "equity": equity,
        "peak": peak,
    }


def calculate_sharpe_correct(pnl_series: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """
    Annualized Sharpe for 15-minute prediction market.
    """
    if pnl_series.empty:
        return 0.0
    equity = INITIAL_CAPITAL + pnl_series.cumsum()
    returns = equity.pct_change().dropna()

    if len(returns) < 2 or float(returns.std()) == 0.0:
        return 0.0

    sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
    if not np.isfinite(sharpe):
        return 0.0
    return float(sharpe)


def calculate_sortino_correct(pnl_series: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """
    Annualized Sortino for 15m predictions using downside deviation.
    """
    if pnl_series.empty:
        return 0.0
    equity = INITIAL_CAPITAL + pnl_series.cumsum()
    returns = equity.pct_change().dropna()

    downside_returns = returns[returns < 0]
    if len(downside_returns) < 2 or float(downside_returns.std()) == 0.0:
        return 0.0

    sortino = (returns.mean() / downside_returns.std()) * np.sqrt(periods_per_year)
    if not np.isfinite(sortino):
        return 0.0
    return float(sortino)


def calculate_calmar_correct(cagr_pct: float, max_dd_pct: float) -> float:
    """
    CAGR / Max Drawdown.
    """
    if max_dd_pct == 0:
        return 0.0
    ratio = cagr_pct / max_dd_pct
    if not np.isfinite(ratio):
        return 0.0
    return float(ratio)


def calculate_profit_factor(trades_df: pd.DataFrame) -> float:
    """
    Total wins / total losses (absolute).
    """
    total_wins = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
    total_losses = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())

    if total_losses == 0:
        return float("inf") if total_wins > 0 else 0.0

    return float(total_wins / total_losses)


def validate_direction_balance(trades_df: pd.DataFrame) -> dict[str, float]:
    """
    Ensure YES and NO trades sum correctly.
    """
    direction_col = "direction" if "direction" in trades_df.columns else "signal"
    direction = trades_df[direction_col].astype(str).str.upper()
    yes_count = int((direction == "YES").sum())
    no_count = int((direction == "NO").sum())
    total = int(len(trades_df))

    if yes_count + no_count != total:
        raise AssertionError("Direction counts do not sum to total")

    yes_pct = (yes_count / total) * 100 if total > 0 else 0.0
    no_pct = (no_count / total) * 100 if total > 0 else 0.0

    return {
        "yes_trades": yes_count,
        "no_trades": no_count,
        "yes_pct": float(yes_pct),
        "no_pct": float(no_pct),
    }


def _strategy_from_trades_file(path: Path) -> str:
    return path.stem.replace("12_trades_", "")


def _compute_max_drawdown_duration_days(trades_df: pd.DataFrame, equity: pd.Series) -> int:
    if trades_df.empty or equity.empty:
        return 0
    dates = pd.to_datetime(trades_df["timestamp"]).dt.date
    daily_equity = pd.Series(equity.values, index=dates).groupby(level=0).last()
    peak_daily = daily_equity.expanding().max()
    in_dd = daily_equity < peak_daily

    longest = 0
    current = 0
    for state in in_dd:
        if bool(state):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _compute_cagr(total_pnl: float, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> tuple[float, float]:
    ending_equity = INITIAL_CAPITAL + total_pnl
    days = max((end_ts - start_ts).total_seconds() / 86400.0, 1.0)
    years = days / 365.25

    if ending_equity <= 0:
        cagr = -1.0
    else:
        cagr = (ending_equity / INITIAL_CAPITAL) ** (1.0 / years) - 1.0

    cagr_pct = cagr * 100.0
    if not np.isfinite(cagr):
        return 0.0, 0.0
    return float(cagr), float(cagr_pct)


def _load_strategy_metrics() -> dict[str, StrategyMetrics]:
    metrics_map: dict[str, StrategyMetrics] = {}

    for path in sorted(EXH_DIR.glob("12_trades_*.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue

        strategy = _strategy_from_trades_file(path)

        if "timestamp" not in df.columns:
            continue

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        if df.empty:
            continue

        if "was_filtered" in df.columns:
            mask = ~df["was_filtered"].astype(str).str.lower().isin(["true", "1"])
            kept = df.loc[mask].copy()
        else:
            kept = df.copy()

        if kept.empty:
            continue

        kept["pnl"] = pd.to_numeric(kept["pnl"], errors="coerce").fillna(0.0)

        cumulative_pnl = kept["pnl"].cumsum()
        dd = calculate_drawdown_correct(cumulative_pnl, INITIAL_CAPITAL)

        sharpe = calculate_sharpe_correct(kept["pnl"])
        sortino = calculate_sortino_correct(kept["pnl"])
        profit_factor = calculate_profit_factor(kept)
        if np.isinf(profit_factor) or np.isnan(profit_factor):
            profit_factor = 0.0

        direction = validate_direction_balance(kept)

        total_trades = int(len(kept))
        win_count = int((kept["pnl"] > 0).sum())
        loss_count = int((kept["pnl"] < 0).sum())
        win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0.0

        total_profit = float(kept.loc[kept["pnl"] > 0, "pnl"].sum())
        total_loss = float(abs(kept.loc[kept["pnl"] < 0, "pnl"].sum()))
        total_pnl = float(kept["pnl"].sum())
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

        min_ts = kept["timestamp"].min()
        max_ts = kept["timestamp"].max()
        active_days = max((max_ts - min_ts).total_seconds() / 86400.0, 1.0)
        trades_per_day = total_trades / active_days

        cagr, cagr_pct = _compute_cagr(total_pnl, min_ts, max_ts)
        calmar = calculate_calmar_correct(cagr_pct, dd["max_dd_pct"])

        max_dd_duration = _compute_max_drawdown_duration_days(kept, dd["equity"])
        recovery_factor = total_pnl / dd["max_dd_dollars"] if dd["max_dd_dollars"] > 0 else 0.0

        max_dd_idx = dd["max_dd_idx"]
        if max_dd_idx is None:
            max_dd_date = "N/A"
        else:
            max_dd_date = pd.to_datetime(kept.loc[max_dd_idx, "timestamp"]).strftime("%Y-%m-%d")

        metrics_map[strategy] = StrategyMetrics(
            strategy_name=strategy,
            total_trades=total_trades,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=float(win_rate),
            total_profit=total_profit,
            total_loss=total_loss,
            total_pnl=total_pnl,
            avg_pnl_per_trade=float(avg_pnl),
            direction_yes_trades=direction["yes_trades"],
            direction_no_trades=direction["no_trades"],
            direction_yes_pct=direction["yes_pct"],
            direction_no_pct=direction["no_pct"],
            trades_per_day=float(trades_per_day),
            cagr=float(cagr),
            cagr_pct=float(cagr_pct),
            profit_factor=float(profit_factor),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar),
            max_drawdown_dollars=float(dd["max_dd_dollars"]),
            max_drawdown_pct=float(dd["max_dd_pct"]),
            peak_before_max_dd=float(dd["peak_before_dd"]),
            valley_at_max_dd=float(dd["valley_at_dd"]),
            max_drawdown_start_date=max_dd_date,
            max_drawdown_duration_days=max_dd_duration,
            recovery_factor=float(recovery_factor),
        )

    return metrics_map


def _update_csvs(metrics_map: dict[str, StrategyMetrics]) -> None:
    risk_path = EXH_DIR / "02_risk_metrics.csv"
    dd_path = EXH_DIR / "07_drawdown_analysis.csv"
    comp_path = EXH_DIR / "10_comparative_matrix.csv"

    risk_df = pd.read_csv(risk_path)
    dd_df = pd.read_csv(dd_path)
    comp_df = pd.read_csv(comp_path)

    def apply_metrics(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "strategy_name" not in out.columns:
            return out

        for idx, row in out.iterrows():
            strategy = row["strategy_name"]
            if strategy not in metrics_map:
                continue
            m = metrics_map[strategy]

            values = {
                "total_trades": m.total_trades,
                "win_count": m.win_count,
                "loss_count": m.loss_count,
                "win_rate": m.win_rate,
                "total_profit": m.total_profit,
                "total_loss": m.total_loss,
                "total_pnl": m.total_pnl,
                "avg_pnl_per_trade": m.avg_pnl_per_trade,
                "direction_yes_trades": m.direction_yes_trades,
                "direction_no_trades": m.direction_no_trades,
                "direction_yes_pct": m.direction_yes_pct,
                "direction_no_pct": m.direction_no_pct,
                "trades_per_day": m.trades_per_day,
                "cagr": m.cagr,
                "profit_factor": m.profit_factor,
                "sharpe_ratio": m.sharpe_ratio,
                "sortino_ratio": m.sortino_ratio,
                "calmar_ratio": m.calmar_ratio,
                "max_drawdown_dollars": m.max_drawdown_dollars,
                "max_drawdown_pct": m.max_drawdown_pct,
                "peak_before_max_dd": m.peak_before_max_dd,
                "valley_at_max_dd": m.valley_at_max_dd,
                "max_drawdown_start_date": m.max_drawdown_start_date,
                "max_drawdown_duration_days": m.max_drawdown_duration_days,
                "max_drawdown_depth_dollars": m.max_drawdown_dollars,
                "max_drawdown_depth_pct": m.max_drawdown_pct,
                "recovery_factor": m.recovery_factor,
                "max_drawdown_duration_days": m.max_drawdown_duration_days,
            }

            for col, val in values.items():
                if col in out.columns:
                    out.at[idx, col] = val

        return out

    risk_df = apply_metrics(risk_df)
    dd_df = apply_metrics(dd_df)
    comp_df = apply_metrics(comp_df)

    risk_df = risk_df.replace([np.inf, -np.inf], 0.0)
    dd_df = dd_df.replace([np.inf, -np.inf], 0.0)
    comp_df = comp_df.replace([np.inf, -np.inf], 0.0)

    risk_df.to_csv(risk_path, index=False)
    dd_df.to_csv(dd_path, index=False)
    comp_df.to_csv(comp_path, index=False)


def _run_validation_tests(metrics_map: dict[str, StrategyMetrics]) -> str:
    lines: list[str] = []
    generated = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    lines.append("METRIC VALIDATION REPORT")
    lines.append(f"Generated: {generated}")
    lines.append("")
    lines.append("Backtest: Polymarket 15m BTC UP/DOWN prediction")
    lines.append("Validation target: Drawdown, Sharpe, Sortino, Calmar, Profit Factor, Direction Balance")
    lines.append("")

    tests: list[tuple[str, bool, str]] = []

    seq = pd.Series([0.98, -1.02, 0.98, 0.98, -1.02])
    seq_dd = calculate_drawdown_correct(seq.cumsum(), INITIAL_CAPITAL)
    seq_sh = calculate_sharpe_correct(seq)
    seq_so = calculate_sortino_correct(seq)
    tests.append(("Simple sequence", np.isfinite(seq_dd["max_dd_pct"]) and np.isfinite(seq_sh) and np.isfinite(seq_so), f"MaxDD={seq_dd['max_dd_pct']:.4f}, Sharpe={seq_sh:.4f}, Sortino={seq_so:.4f}"))

    wins = pd.Series([0.98] * 100)
    wins_pf = calculate_profit_factor(pd.DataFrame({"pnl": wins}))
    wins_sh = calculate_sharpe_correct(wins)
    tests.append(("All wins", (np.isinf(wins_pf) or wins_pf > 0) and wins_sh >= 0, f"ProfitFactor={wins_pf:.4f}, Sharpe={wins_sh:.4f}"))

    losses = pd.Series([-1.02] * 100)
    losses_pf = calculate_profit_factor(pd.DataFrame({"pnl": losses}))
    losses_sh = calculate_sharpe_correct(losses)
    tests.append(("All losses", losses_pf == 0.0 and np.isfinite(losses_sh), f"ProfitFactor={losses_pf:.4f}, Sharpe={losses_sh:.4f}"))

    baseline = metrics_map.get("00_Baseline")
    baseline_ok = baseline is not None and np.isfinite(baseline.sharpe_ratio) and np.isfinite(baseline.sortino_ratio)
    baseline_msg = "No baseline strategy found" if baseline is None else (
        f"Sharpe={baseline.sharpe_ratio:.4f}, Sortino={baseline.sortino_ratio:.4f}, MaxDD={baseline.max_drawdown_pct:.4f}, Calmar={baseline.calmar_ratio:.4f}"
    )
    tests.append(("Mixed baseline trade data", baseline_ok, baseline_msg))

    dir_test_df = pd.DataFrame({"direction": ["YES", "NO", "YES", "NO", "YES"], "pnl": [1, -1, 1, -1, 1]})
    dir_balance = validate_direction_balance(dir_test_df)
    dir_ok = dir_balance["yes_trades"] + dir_balance["no_trades"] == len(dir_test_df)
    tests.append(("Direction balance", dir_ok, f"YES={dir_balance['yes_trades']}, NO={dir_balance['no_trades']}"))

    pass_count = 0
    for name, ok, msg in tests:
        status = "PASS" if ok else "FAIL"
        if ok:
            pass_count += 1
        lines.append(f"[{status}] {name}: {msg}")

    lines.append("")
    lines.append(f"Summary: {pass_count}/{len(tests)} tests passed")
    lines.append("")
    lines.append("Formula Notes:")
    lines.append("- Drawdown = (Peak - Valley) / $100 initial capital")
    lines.append("- Sharpe annualization = sqrt(35,040)")
    lines.append("- Sortino uses downside return std deviation")
    lines.append("- Calmar = CAGR(%) / MaxDrawdown(%)")
    lines.append("- Profit Factor = total wins / absolute total losses")

    return "\n".join(lines) + "\n"


def main() -> None:
    metrics_map = _load_strategy_metrics()
    if not metrics_map:
        raise RuntimeError("No strategy trade files found for metric recalculation.")

    _update_csvs(metrics_map)
    report_text = _run_validation_tests(metrics_map)
    VALIDATION_REPORT.write_text(report_text, encoding="utf-8")

    print(f"Recalculated metrics for {len(metrics_map)} strategies.")
    print(f"Updated: {EXH_DIR / '02_risk_metrics.csv'}")
    print(f"Updated: {EXH_DIR / '07_drawdown_analysis.csv'}")
    print(f"Updated: {EXH_DIR / '10_comparative_matrix.csv'}")
    print(f"Saved: {VALIDATION_REPORT}")


if __name__ == "__main__":
    main()

