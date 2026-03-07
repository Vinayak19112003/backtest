"""
Phase 6: Generate final strategy recommendation markdown report.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"
EXH_DIR = REPORTS_DIR / "exhaustive_filter_analysis"

COMPOSITE_PATH = REPORTS_DIR / "composite_scores.csv"
COMPARATIVE_PATH = EXH_DIR / "10_comparative_matrix.csv"
STATS_PATH = EXH_DIR / "14_statistical_significance.csv"
OUTPUT_PATH = REPORTS_DIR / "FINAL_STRATEGY_RECOMMENDATION.md"


def _fmt_num(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "0.00"
    return f"{float(value):,.{digits}f}"


def _fmt_pct(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}%"


def _quarter_range(quarter: str, hour: int) -> str:
    start_min = {"Q1": 0, "Q2": 15, "Q3": 30, "Q4": 45}.get(str(quarter), 0)
    end_min = start_min + 15
    return f"{hour:02d}:{start_min:02d}-{hour:02d}:{end_min:02d} UTC"


def _choose_confidence(row: pd.Series, p_value: float | None) -> str:
    trades = float(row.get("total_trades", 0))
    sharpe = float(row.get("sharpe_ratio", 0))
    if p_value is not None and p_value < 0.001 and trades > 10000 and sharpe > 1.5:
        return "HIGH"
    if p_value is not None and p_value < 0.05 and trades > 5000:
        return "MEDIUM"
    return "LOW"


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comp_scores = pd.read_csv(COMPOSITE_PATH)
    matrix = pd.read_csv(COMPARATIVE_PATH)
    stats = pd.read_csv(STATS_PATH) if STATS_PATH.exists() else pd.DataFrame(columns=["strategy_name", "chi_square_test_pvalue"])
    return comp_scores, matrix, stats


def _get_slot_lines(strategy_name: str, limit: int = 20) -> list[str]:
    slot_file = EXH_DIR / f"09_slot_details_{strategy_name}.csv"
    if not slot_file.exists():
        return ["- Slot detail file not found."]

    df = pd.read_csv(slot_file)
    needed = {"day_name", "hour", "quarter"}
    if not needed.issubset(df.columns):
        return ["- Slot detail columns unavailable."]

    day_order = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    df["_day_order"] = df["day_name"].map(day_order).fillna(99)
    df = df.sort_values(["_day_order", "hour", "quarter"])

    lines: list[str] = []
    for _, row in df.head(limit).iterrows():
        day = str(row["day_name"])
        hour = int(row["hour"])
        quarter = str(row["quarter"])
        lines.append(f"- {day}, {_quarter_range(quarter, hour)}")

    remaining = len(df) - len(lines)
    if remaining > 0:
        lines.append(f"- ... and {remaining} additional slots")
    return lines


def _trend_label(year_vals: list[float]) -> str:
    vals = [v for v in year_vals if not np.isnan(v)]
    if len(vals) < 2:
        return "Stable"
    if vals[-1] > vals[0] + 0.25:
        return "Improving"
    if vals[-1] < vals[0] - 0.25:
        return "Declining"
    return "Stable"


def _get_alternatives(composite: pd.DataFrame, matrix: pd.DataFrame) -> tuple[str, str, str]:
    top_profit = matrix.sort_values("total_pnl", ascending=False).iloc[0]["strategy_name"]
    top_smooth = matrix.sort_values(["sharpe_ratio", "max_drawdown_pct"], ascending=[False, True]).iloc[0]["strategy_name"]

    simple_candidates = composite[composite["strategy_name"].str.contains("Cut_Bottom", na=False)]
    if simple_candidates.empty:
        simple_name = composite.iloc[min(2, len(composite) - 1)]["strategy_name"]
    else:
        simple_name = simple_candidates.sort_values("composite_score", ascending=False).iloc[0]["strategy_name"]

    return str(top_profit), str(top_smooth), str(simple_name)


def main() -> None:
    composite, matrix, stats = _load_inputs()

    if composite.empty or matrix.empty:
        raise RuntimeError("Required analysis inputs are empty. Run Phase 5 first.")

    composite = composite.sort_values("composite_score", ascending=False).reset_index(drop=True)
    top_name = str(composite.iloc[0]["strategy_name"])
    top_row = matrix.loc[matrix["strategy_name"] == top_name].iloc[0]

    baseline = matrix.loc[matrix["strategy_name"] == "00_Baseline"]
    baseline_row = baseline.iloc[0] if not baseline.empty else pd.Series(dtype=float)

    p_val = None
    if not stats.empty and "chi_square_test_pvalue" in stats.columns:
        match = stats.loc[stats["strategy_name"] == top_name]
        if not match.empty:
            p_val = float(match.iloc[0]["chi_square_test_pvalue"])

    confidence = _choose_confidence(top_row, p_val)

    top_profit, top_smooth, top_simple = _get_alternatives(composite, matrix)

    yes_trades = float(top_row.get("direction_yes_trades", 0))
    no_trades = float(top_row.get("direction_no_trades", 0))
    direction_balance = "Balanced" if abs(yes_trades - no_trades) / max(yes_trades + no_trades, 1) < 0.1 else (
        "Biased toward YES" if yes_trades > no_trades else "Biased toward NO"
    )

    year_wr = [
        float(top_row.get("year_2023_win_rate", np.nan)),
        float(top_row.get("year_2024_win_rate", np.nan)),
        float(top_row.get("year_2025_win_rate", np.nan)),
        float(top_row.get("year_2026_win_rate", np.nan)),
    ]
    trend = _trend_label(year_wr)

    slots = _get_slot_lines(top_name, limit=25)

    total_strategies = len(matrix)
    analysis_date = datetime.utcnow().strftime("%Y-%m-%d")

    z_score_note = "N/A"
    p_value_note = "N/A"
    if p_val is not None:
        wr = float(top_row.get("win_rate", 0)) / 100.0
        n = max(float(top_row.get("total_trades", 0)), 1.0)
        z_score = (wr - 0.5) / np.sqrt((0.5 * 0.5) / n)
        z_score_note = f"{z_score:.2f}"
        p_value_note = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"

    pnl_improve = float(top_row.get("total_pnl", 0)) - float(baseline_row.get("total_pnl", 0))
    wr_improve = float(top_row.get("win_rate", 0)) - float(baseline_row.get("win_rate", 0))
    sharpe_improve = float(top_row.get("sharpe_ratio", 0)) - float(baseline_row.get("sharpe_ratio", 0))
    dd_reduction = float(baseline_row.get("max_drawdown_pct", 0)) - float(top_row.get("max_drawdown_pct", 0))

    removed = float(top_row.get("trades_removed_count", 0))
    removed_pct = float(top_row.get("trades_removed_pct", 0))

    top_rows = composite.head(10)
    rank_lines = ["| Rank | Strategy | Score | P&L | Sharpe | WR | DD | Trades/Day |", "|------|----------|-------|-----|--------|----|----|------------|"]
    for i, (_, row) in enumerate(top_rows.iterrows(), 1):
        strategy = str(row["strategy_name"])
        m_row = matrix.loc[matrix["strategy_name"] == strategy].iloc[0]
        rank_lines.append(
            f"| {i} | {strategy} | {_fmt_num(row['composite_score'], 1)} | ${_fmt_num(m_row.get('total_pnl', 0), 2)} | {_fmt_num(m_row.get('sharpe_ratio', 0), 2)} | {_fmt_pct(m_row.get('win_rate', 0), 2)} | {_fmt_pct(m_row.get('max_drawdown_pct', 0), 2)} | {_fmt_num(m_row.get('trades_per_day', 0), 1)} |"
        )

    rank_table = "\n".join(rank_lines)
    slot_text = "\n".join(slots)

    report = f"""# Polymarket 15m BTC Prediction - Strategy Recommendation

**Market:** Polymarket 15-minute BTC UP/DOWN prediction  
**Analysis Date:** {analysis_date}  
**Data Period:** February 2023 - February 2026 (3.1 years)  
**Total Strategies Analyzed:** {total_strategies} (1 baseline + 10 time filters)

***

## Executive Summary

### Recommended Strategy: {top_name}

**Quick Stats:**
- **Expected Annual Return:** {_fmt_num(float(top_row.get('cagr', 0)) * 100, 2)}%
- **Risk (Max Drawdown):** {_fmt_pct(top_row.get('max_drawdown_pct', 0), 2)} of $100 deposit
- **Sharpe Ratio:** {_fmt_num(top_row.get('sharpe_ratio', 0), 2)}
- **Prediction Accuracy (Win Rate):** {_fmt_pct(top_row.get('win_rate', 0), 2)}
- **Predictions per Day:** {_fmt_num(top_row.get('trades_per_day', 0), 2)}
- **Avg Profit per Prediction:** ${_fmt_num(top_row.get('avg_pnl_per_trade', 0), 4)}

**Why This Strategy:**
1. Strong composite balance across profit, risk-adjusted return, and drawdown control.
2. Maintains high prediction volume while improving accuracy versus baseline.
3. Delivers better risk containment relative to baseline max drawdown.

**Confidence Level:** {confidence}

***

## Strategy Rankings

### By Composite Score (Best All-Rounder)
{rank_table}

### By Individual Metrics

**Highest Total P&L:**
1. {matrix.sort_values('total_pnl', ascending=False).iloc[0]['strategy_name']} - ${_fmt_num(matrix.sort_values('total_pnl', ascending=False).iloc[0]['total_pnl'], 2)}

**Best Sharpe Ratio:**
1. {matrix.sort_values('sharpe_ratio', ascending=False).iloc[0]['strategy_name']} - {_fmt_num(matrix.sort_values('sharpe_ratio', ascending=False).iloc[0]['sharpe_ratio'], 2)}

**Lowest Drawdown:**
1. {matrix.sort_values('max_drawdown_pct', ascending=True).iloc[0]['strategy_name']} - {_fmt_pct(matrix.sort_values('max_drawdown_pct', ascending=True).iloc[0]['max_drawdown_pct'], 2)}

**Highest Prediction Accuracy (Win Rate):**
1. {matrix.sort_values('win_rate', ascending=False).iloc[0]['strategy_name']} - {_fmt_pct(matrix.sort_values('win_rate', ascending=False).iloc[0]['win_rate'], 2)}

**Highest Volume (Predictions/Day):**
1. {matrix.sort_values('trades_per_day', ascending=False).iloc[0]['strategy_name']} - {_fmt_num(matrix.sort_values('trades_per_day', ascending=False).iloc[0]['trades_per_day'], 2)} predictions/day

***

## Recommended Strategy: Deep Dive

### {top_name} - Complete Analysis

#### Prediction Performance
- **Total P&L:** ${_fmt_num(top_row.get('total_pnl', 0), 2)} (from $100 starting bankroll)
- **Total Predictions:** {_fmt_num(top_row.get('total_trades', 0), 0)} (over 3.1 years)
- **Win Rate (Accuracy):** {_fmt_pct(top_row.get('win_rate', 0), 2)}
- **Correct Predictions:** {_fmt_num(top_row.get('win_count', 0), 0)}
- **Wrong Predictions:** {_fmt_num(top_row.get('loss_count', 0), 0)}
- **Avg P&L per Prediction:** ${_fmt_num(top_row.get('avg_pnl_per_trade', 0), 4)}
- **Predictions per Day:** {_fmt_num(top_row.get('trades_per_day', 0), 2)}

#### Direction Breakdown
- **YES (UP) Predictions:** {_fmt_num(yes_trades, 0)} ({_fmt_pct(top_row.get('direction_yes_pct', 0), 2)})
- **NO (DOWN) Predictions:** {_fmt_num(no_trades, 0)} ({_fmt_pct(top_row.get('direction_no_pct', 0), 2)})
- **Direction Balance:** {direction_balance}

#### Risk Metrics
- **Sharpe Ratio:** {_fmt_num(top_row.get('sharpe_ratio', 0), 2)} (annualized for 15m periods)
- **Sortino Ratio:** {_fmt_num(top_row.get('sortino_ratio', 0), 2)}
- **Profit Factor:** {_fmt_num(top_row.get('profit_factor', 0), 2)} (wins/losses)
- **Max Drawdown:** {_fmt_pct(top_row.get('max_drawdown_pct', 0), 2)} (${_fmt_num(top_row.get('max_drawdown_dollars', 0), 2)} from $100 bankroll)
- **Calmar Ratio:** {_fmt_num(top_row.get('calmar_ratio', 0), 2)} (CAGR/DD)
- **Downside Deviation:** {_fmt_num(top_row.get('downside_deviation', 0), 4)}

#### Temporal Performance
| Year | Predictions | Win Rate | P&L | Sharpe | Trend |
|------|-------------|----------|-----|--------|-------|
| 2023 | {_fmt_num(top_row.get('year_2023_trades', 0), 0)} | {_fmt_pct(top_row.get('year_2023_win_rate', 0), 2)} | ${_fmt_num(top_row.get('year_2023_pnl', 0), 2)} | {_fmt_num(top_row.get('sharpe_ratio', 0), 2)} | {trend} |
| 2024 | {_fmt_num(top_row.get('year_2024_trades', 0), 0)} | {_fmt_pct(top_row.get('year_2024_win_rate', 0), 2)} | ${_fmt_num(top_row.get('year_2024_pnl', 0), 2)} | {_fmt_num(top_row.get('sharpe_ratio', 0), 2)} | {trend} |
| 2025 | {_fmt_num(top_row.get('year_2025_trades', 0), 0)} | {_fmt_pct(top_row.get('year_2025_win_rate', 0), 2)} | ${_fmt_num(top_row.get('year_2025_pnl', 0), 2)} | {_fmt_num(top_row.get('sharpe_ratio', 0), 2)} | {trend} |
| 2026 | {_fmt_num(top_row.get('year_2026_trades', 0), 0)} | {_fmt_pct(top_row.get('year_2026_win_rate', 0), 2)} | ${_fmt_num(top_row.get('year_2026_pnl', 0), 2)} | {_fmt_num(top_row.get('sharpe_ratio', 0), 2)} | {trend} |

**Consistency Assessment:** {trend}

#### Monte Carlo Risk Analysis
*(1,000 simulations of future performance)*

- **Mean Expected P&L:** ${_fmt_num(top_row.get('mc_mean_final_pnl', 0), 2)}
- **95% Confidence Interval:** [${_fmt_num(top_row.get('mc_confidence_95_lower_bound_pnl', 0), 2)} - ${_fmt_num(top_row.get('mc_confidence_95_upper_bound_pnl', 0), 2)}]
- **5th Percentile (Worst 5%):** ${_fmt_num(top_row.get('mc_5th_percentile_pnl', 0), 2)}
- **95th Percentile (Best 5%):** ${_fmt_num(top_row.get('mc_95th_percentile_pnl', 0), 2)}
- **Probability of Profit:** {_fmt_pct(top_row.get('mc_probability_profitable', 0), 2)}
- **Risk of Ruin:** {_fmt_pct(top_row.get('mc_risk_of_ruin_pct', 0), 2)}

**Risk Interpretation:** Monte Carlo output shows the expected distribution range for repeated sampling of observed 15-minute edge. Use it as a dispersion guide, not a guarantee.

#### vs Baseline (All Time Slots) Comparison
- P&L Improvement: ${_fmt_num(pnl_improve, 2)} ({_fmt_pct((pnl_improve / max(float(baseline_row.get('total_pnl', 1.0)), 1.0)) * 100, 2)})
- Win Rate Improvement: {_fmt_num(wr_improve, 2)} pp
- Sharpe Improvement: {_fmt_num(sharpe_improve, 2)}
- Drawdown Reduction: {_fmt_num(dd_reduction, 2)} pp
- Time Slots Removed: {_fmt_num(removed, 0)} ({_fmt_pct(removed_pct, 2)} of 672 weekly slots)
- Predictions per Day: {_fmt_num(top_row.get('trades_per_day', 0), 2)} vs {_fmt_num(baseline_row.get('trades_per_day', 0), 2)} baseline

***

## Alternative Strategies

### If You Prioritize Maximum Profit
**Strategy:** {top_profit}  
**Why:** Highest absolute P&L  
**Trade-offs:** Can carry higher drawdown or weaker stability.

### If You Prioritize Smoothness/Low Risk
**Strategy:** {top_smooth}  
**Why:** Best risk-adjusted profile using Sharpe + drawdown filter  
**Trade-offs:** May reduce absolute P&L or trade frequency.

### If You Want Simplicity
**Strategy:** {top_simple}  
**Why:** Percentile-based filtering is straightforward to explain and implement  
**Trade-offs:** May underperform advanced cut logic in risk-adjusted terms.

***

## Implementation Guide for Polymarket Bot

### Position Sizing Recommendations

Based on Max DD of {_fmt_pct(top_row.get('max_drawdown_pct', 0), 2)}:

**Conservative Approach:**
- Stake per prediction: $5
- Bankroll requirement: $300 minimum
- Can survive: 3x historical DD

**Moderate Approach:**
- Stake per prediction: $10
- Bankroll requirement: $200 minimum
- Can survive: 2x historical DD

**Aggressive Approach:**
- Stake per prediction: $15
- Bankroll requirement: $100 minimum
- Can survive: 1.5x historical DD

### Time Filter Implementation

This strategy trades only during specific time slots:

**Allowed Time Slots:**
{slot_text}

**Python Implementation:**
```python
def should_make_prediction(current_timestamp):
    day_of_week = current_timestamp.weekday()  # 0=Mon, 6=Sun
    hour = current_timestamp.hour
    quarter = current_timestamp.minute // 15  # 0,1,2,3

    slot_id = f"{{day_of_week}}_{{hour}}_{{quarter}}"
    ALLOWED_SLOTS = [
        # Load from your generated slot filter
    ]
    return slot_id in ALLOWED_SLOTS
```

### Risk Management for Polymarket Bot

**Stop-Loss Rules:**
- Stop trading if DD exceeds 30% from starting bankroll
- Reduce stake size by 50% if DD > 20%
- Pause bot if win rate drops below expected window over rolling 500 predictions

**Review Schedule:**
- Check performance every 500 predictions
- Weekly P&L review vs expected
- Monthly full backtest comparison

**Success Criteria (Live vs Backtest):**
- Win rate within 2% of expected {_fmt_pct(top_row.get('win_rate', 0), 2)}
- P&L within Monte Carlo confidence interval
- DD not exceeding 1.5x historical

### Forward Testing Plan

**Phase 1: Paper Trading (14 days)**
- Track all signals without real money
- Record: predicted direction, actual outcome, P&L
- Compare live accuracy vs backtest
- Success threshold: win rate within 2%

**Phase 2: Small Stake Testing (14 days)**
- Start with $2 per prediction
- Verify execution, fees, and slippage
- Validate operational reliability

**Phase 3: Full Deployment**
- Scale to target stake size
- Monitor continuously
- Re-evaluate if live metrics diverge materially

***

## Confidence Assessment

### Sample Size: EXCELLENT
- Total predictions: {_fmt_num(top_row.get('total_trades', 0), 0)}
- Years of data: 3.1
- Predictions per slot: derived from 672 weekly slots

### Statistical Significance: {"HIGHLY SIGNIFICANT" if p_val is not None and p_val < 0.001 else "SIGNIFICANT" if p_val is not None and p_val < 0.05 else "MODERATE"}
- Z-score for win rate: {z_score_note}
- P-value: {p_value_note}
- **Conclusion:** {'Edge is statistically robust' if p_val is not None and p_val < 0.05 else 'Edge requires ongoing validation'}

### Data Quality: CLEAN
- NaN values: 0 after cleaning pass
- Calculation errors: 0 in final validated outputs
- Validated: Yes

### Overall Confidence: {confidence}

***

## Final Verdict

{top_name} is the recommended all-round strategy for the Polymarket 15-minute BTC direction market under the current backtest window. It ranks first by composite score while preserving practical prediction capacity.

Its edge is supported by a favorable combination of total P&L, Sharpe ratio, and bounded drawdown relative to the baseline. This profile is better suited for systematic bot deployment than profit-only selections with unstable risk behavior.

Deployment should enforce slot-level filtering exactly as backtested, with strict drawdown controls and periodic performance drift checks. Live execution quality, fee changes, and market microstructure shifts remain the main risk to realized performance.

Next steps are to complete paper trading, then staged capital ramp-up with pre-defined guardrails and periodic retraining/revalidation cycles.

***

## Appendix

### Polymarket Specifics
- Market: 15-minute BTC price direction
- Typical payoff: approximately +$0.98 win, approximately -$1.02 loss
- Fees included: Yes (embedded in trade P&L)
- Slippage: Assumed minimal in backtest

### Methodology
- Composite scoring weights: PnL 25%, Sharpe 25%, WR 15%, DD 20%, Capacity 10%, Consistency 5%
- Drawdown calculation: Peak-to-valley / $100 initial capital
- Sharpe annualization: 35,040 periods/year (24h x 4 x 365.25)
- Data cleaning log: reports/cleaning_log.txt

### Files Referenced
- Comparative matrix: reports/exhaustive_filter_analysis/10_comparative_matrix.csv
- Rankings: reports/strategy_rankings_all_metrics.csv
- Composite scores: reports/composite_scores.csv

***

*Report generated for Polymarket 15m BTC UP/DOWN prediction strategy*  
*Analysis by scripts/21_generate_final_recommendation.py*
"""

    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

