"""
Phase 1: Data quality audit for Polymarket 15m BTC UP/DOWN reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"
AUDIT_PATH = REPORTS_DIR / "data_quality_audit.csv"
SUMMARY_PATH = REPORTS_DIR / "data_quality_summary.txt"


@dataclass
class AuditIssue:
    file_path: str
    issue_type: str
    column_name: str
    affected_rows: int
    severity: str
    example_value: str
    suggested_fix: str


REQUIRED_COLUMNS_BY_FILE: dict[str, list[str]] = {
    "reports/exhaustive_filter_analysis/01_basic_metrics.csv": [
        "strategy_name",
        "total_trades",
        "win_count",
        "loss_count",
        "win_rate",
        "total_pnl",
        "direction_yes_trades",
        "direction_no_trades",
    ],
    "reports/exhaustive_filter_analysis/02_risk_metrics.csv": [
        "strategy_name",
        "profit_factor",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown_pct",
    ],
    "reports/exhaustive_filter_analysis/07_drawdown_analysis.csv": [
        "strategy_name",
        "max_drawdown_duration_days",
        "max_drawdown_depth_dollars",
        "max_drawdown_depth_pct",
    ],
    "reports/exhaustive_filter_analysis/10_comparative_matrix.csv": [
        "strategy_name",
        "total_trades",
        "win_rate",
        "total_pnl",
        "sharpe_ratio",
        "max_drawdown_pct",
    ],
}

REQUIRED_TRADE_COLUMNS = [
    "trade_id",
    "timestamp",
    "signal",
    "win",
    "pnl",
    "fees",
]

COUNT_COLUMNS = {
    "trade_id",
    "total_trades",
    "trades_removed_count",
    "win_count",
    "loss_count",
    "direction_yes_trades",
    "direction_no_trades",
    "drawdown_5pct_count",
    "drawdown_10pct_count",
    "drawdown_20pct_count",
    "days_underwater",
    "max_consecutive_wins",
    "max_consecutive_losses",
    "busiest_day_trades",
    "quietest_day_trades",
    "zero_days_count",
    "year_2023_trades",
    "year_2024_trades",
    "year_2025_trades",
    "year_2026_trades",
}

BOUNDED_PCT_COLUMNS = {
    "win_rate",
    "loss_rate",
    "direction_yes_pct",
    "direction_no_pct",
    "year_2023_win_rate",
    "year_2024_win_rate",
    "year_2025_win_rate",
    "year_2026_win_rate",
    "quarterly_win_rate_q1",
    "quarterly_win_rate_q2",
    "quarterly_win_rate_q3",
    "quarterly_win_rate_q4",
    "mc_probability_profitable",
    "mc_risk_of_ruin_pct",
}


def _to_rel(path: Path) -> str:
    return path.relative_to(BASE_DIR).as_posix()


def _safe_str(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    text = str(value)
    return text[:200]


def _add_issue(
    issues: list[AuditIssue],
    file_path: Path,
    issue_type: str,
    column_name: str,
    affected_rows: int,
    severity: str,
    example_value: Any,
    suggested_fix: str,
) -> None:
    issues.append(
        AuditIssue(
            file_path=_to_rel(file_path),
            issue_type=issue_type,
            column_name=column_name,
            affected_rows=int(affected_rows),
            severity=severity,
            example_value=_safe_str(example_value),
            suggested_fix=suggested_fix,
        )
    )


def _check_missing_columns(file_path: Path, columns: list[str], issues: list[AuditIssue]) -> None:
    rel = _to_rel(file_path)
    required = REQUIRED_COLUMNS_BY_FILE.get(rel, [])
    if rel.startswith("reports/exhaustive_filter_analysis/12_trades_"):
        required = REQUIRED_TRADE_COLUMNS
    for col in required:
        if col not in columns:
            _add_issue(
                issues,
                file_path,
                "Missing",
                col,
                1,
                "Critical",
                "missing",
                "Add the required column and regenerate this report from source data.",
            )


def _check_nan_and_inf(df: pd.DataFrame, file_path: Path, issues: list[AuditIssue]) -> None:
    for col in df.columns:
        nan_mask = df[col].isna()
        nan_count = int(nan_mask.sum())
        if nan_count > 0:
            example = df.loc[nan_mask, col].iloc[0]
            _add_issue(
                issues,
                file_path,
                "NaN",
                col,
                nan_count,
                "Warning",
                example,
                "Recalculate this field from source columns or set stable default when denominator is zero.",
            )

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return
    for col in numeric_df.columns:
        arr = numeric_df[col].to_numpy()
        inf_mask = np.isinf(arr)
        inf_count = int(np.sum(inf_mask))
        if inf_count > 0:
            example = arr[np.where(inf_mask)[0][0]]
            _add_issue(
                issues,
                file_path,
                "Infinite",
                col,
                inf_count,
                "Critical",
                example,
                "Replace +/-inf with 0.0 for undefined risk ratios and log denominator-zero condition.",
            )


def _check_types(df: pd.DataFrame, file_path: Path, issues: list[AuditIssue]) -> None:
    for col in df.columns:
        if col in COUNT_COLUMNS:
            series = pd.to_numeric(df[col], errors="coerce")
            bad = series.isna()
            if bad.any():
                _add_issue(
                    issues,
                    file_path,
                    "TypeError",
                    col,
                    int(bad.sum()),
                    "Warning",
                    df.loc[bad, col].iloc[0],
                    "Cast to integer-compatible numeric type.",
                )
                continue
            non_int = ((series % 1).abs() > 1e-9)
            if non_int.any():
                _add_issue(
                    issues,
                    file_path,
                    "TypeError",
                    col,
                    int(non_int.sum()),
                    "Info",
                    series[non_int].iloc[0],
                    "Counts should be integers; round or recalculate.",
                )

        if col == "timestamp" or col.endswith("_date"):
            parsed = pd.to_datetime(df[col], errors="coerce")
            bad = parsed.isna() & df[col].notna()
            if bad.any():
                _add_issue(
                    issues,
                    file_path,
                    "TypeError",
                    col,
                    int(bad.sum()),
                    "Warning",
                    df.loc[bad, col].iloc[0],
                    "Normalize to ISO date format YYYY-MM-DD or full timestamp.",
                )

    if "signal" in df.columns:
        upper = df["signal"].astype(str).str.upper()
        bad = ~upper.isin(["YES", "NO"])
        if bad.any():
            _add_issue(
                issues,
                file_path,
                "TypeError",
                "signal",
                int(bad.sum()),
                "Critical",
                df.loc[bad, "signal"].iloc[0],
                "Restrict signal values to YES/NO.",
            )


def _check_logic(df: pd.DataFrame, file_path: Path, issues: list[AuditIssue]) -> None:
    for col in BOUNDED_PCT_COLUMNS:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = (vals < 0) | (vals > 100)
        bad = bad.fillna(False)
        if bad.any():
            _add_issue(
                issues,
                file_path,
                "LogicError",
                col,
                int(bad.sum()),
                "Critical",
                vals[bad].iloc[0],
                "Keep bounded percentage metrics within [0, 100].",
            )

    for col in COUNT_COLUMNS.intersection(set(df.columns)):
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = vals < 0
        bad = bad.fillna(False)
        if bad.any():
            _add_issue(
                issues,
                file_path,
                "LogicError",
                col,
                int(bad.sum()),
                "Critical",
                vals[bad].iloc[0],
                "Counts cannot be negative; recalculate from source records.",
            )

    if {"win_count", "loss_count", "total_trades"}.issubset(df.columns):
        win = pd.to_numeric(df["win_count"], errors="coerce")
        loss = pd.to_numeric(df["loss_count"], errors="coerce")
        total = pd.to_numeric(df["total_trades"], errors="coerce")
        bad = (win + loss - total).abs() > 1e-6
        bad = bad.fillna(False)
        if bad.any():
            _add_issue(
                issues,
                file_path,
                "CalcError",
                "total_trades",
                int(bad.sum()),
                "Critical",
                f"win+loss={float((win + loss)[bad].iloc[0]):.4f}, total={float(total[bad].iloc[0]):.4f}",
                "Set total_trades = win_count + loss_count.",
            )

    if {"direction_yes_trades", "direction_no_trades", "total_trades"}.issubset(df.columns):
        yes = pd.to_numeric(df["direction_yes_trades"], errors="coerce")
        no = pd.to_numeric(df["direction_no_trades"], errors="coerce")
        total = pd.to_numeric(df["total_trades"], errors="coerce")
        bad = (yes + no - total).abs() > 1e-6
        bad = bad.fillna(False)
        if bad.any():
            _add_issue(
                issues,
                file_path,
                "CalcError",
                "direction_yes_trades,direction_no_trades",
                int(bad.sum()),
                "Critical",
                f"yes+no={float((yes + no)[bad].iloc[0]):.4f}, total={float(total[bad].iloc[0]):.4f}",
                "Recalculate YES/NO direction counts from trade-level signal data.",
            )

    if {"total_profit", "total_loss", "total_pnl"}.issubset(df.columns):
        total_profit = pd.to_numeric(df["total_profit"], errors="coerce")
        total_loss = pd.to_numeric(df["total_loss"], errors="coerce")
        total_pnl = pd.to_numeric(df["total_pnl"], errors="coerce")
        bad = (total_profit - total_loss - total_pnl).abs() > 1e-4
        bad = bad.fillna(False)
        if bad.any():
            _add_issue(
                issues,
                file_path,
                "CalcError",
                "total_pnl",
                int(bad.sum()),
                "Warning",
                float((total_profit - total_loss - total_pnl)[bad].iloc[0]),
                "Set total_pnl = total_profit - total_loss.",
            )

    if "win" in df.columns and "pnl" in df.columns:
        win_bool = df["win"].astype(str).str.lower().map({"true": True, "false": False})
        pnl = pd.to_numeric(df["pnl"], errors="coerce")
        bad = ((win_bool == True) & (pnl <= 0)) | ((win_bool == False) & (pnl >= 0))
        bad = bad.fillna(False)
        if bad.any():
            _add_issue(
                issues,
                file_path,
                "LogicError",
                "win,pnl",
                int(bad.sum()),
                "Warning",
                f"win={df.loc[bad, 'win'].iloc[0]}, pnl={df.loc[bad, 'pnl'].iloc[0]}",
                "Align win flag with pnl sign (wins > 0, losses < 0).",
            )


def run_audit() -> pd.DataFrame:
    issues: list[AuditIssue] = []
    csv_files = sorted(REPORTS_DIR.rglob("*.csv"))

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:  # noqa: BLE001
            _add_issue(
                issues,
                file_path,
                "TypeError",
                "__file__",
                1,
                "Critical",
                str(exc),
                "Fix CSV formatting and ensure valid delimiter/header.",
            )
            continue

        _check_missing_columns(file_path, list(df.columns), issues)
        _check_nan_and_inf(df, file_path, issues)
        _check_types(df, file_path, issues)
        _check_logic(df, file_path, issues)

    audit_df = pd.DataFrame([issue.__dict__ for issue in issues])
    if audit_df.empty:
        audit_df = pd.DataFrame(
            columns=[
                "file_path",
                "issue_type",
                "column_name",
                "affected_rows",
                "severity",
                "example_value",
                "suggested_fix",
            ]
        )
    return audit_df


def write_summary(audit_df: pd.DataFrame) -> None:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    files_scanned = len(list(REPORTS_DIR.rglob("*.csv")))
    files_with_issues = audit_df["file_path"].nunique() if not audit_df.empty else 0
    total_issues = len(audit_df)

    sev_counts = audit_df["severity"].value_counts().to_dict() if not audit_df.empty else {}
    type_counts = audit_df["issue_type"].value_counts().to_dict() if not audit_df.empty else {}
    status = "CLEAN" if total_issues == 0 else "ISSUES FOUND"

    top_files = (
        audit_df["file_path"].value_counts().head(5).to_dict() if not audit_df.empty else {}
    )

    actions: list[str] = []
    if type_counts.get("Missing", 0) > 0:
        actions.append("Regenerate files with missing required columns before downstream analysis.")
    if type_counts.get("Infinite", 0) > 0:
        actions.append("Replace infinite risk metrics with 0.0 where denominator is zero.")
    if type_counts.get("NaN", 0) > 0:
        actions.append("Recompute NaN fields from source metrics or enforce denominator guards.")
    if type_counts.get("CalcError", 0) > 0:
        actions.append("Fix arithmetic identities (wins/losses totals, direction counts, and PnL equations).")
    if type_counts.get("TypeError", 0) > 0:
        actions.append("Normalize datatypes for dates, numeric fields, and categorical direction labels.")
    if not actions:
        actions.append("No corrective action required.")

    lines = [
        "DATA QUALITY AUDIT SUMMARY",
        f"Generated: {timestamp}",
        "",
        "BACKTEST TYPE: Polymarket 15-minute BTC UP/DOWN prediction",
        "DATA PERIOD: Feb 2023 - Feb 2026 (3.1 years)",
        "",
        f"OVERALL STATUS: {status}",
        "",
        f"FILES SCANNED: {files_scanned}",
        f"FILES WITH ISSUES: {files_with_issues}",
        f"TOTAL ISSUES: {total_issues}",
        "",
        "BY SEVERITY:",
        f"- Critical: {sev_counts.get('Critical', 0)} (blocks analysis)",
        f"- Warning: {sev_counts.get('Warning', 0)} (may affect accuracy)",
        f"- Info: {sev_counts.get('Info', 0)} (minor inconsistencies)",
        "",
        "BY TYPE:",
        f"- NaN values: {type_counts.get('NaN', 0)}",
        f"- Infinite values: {type_counts.get('Infinite', 0)}",
        f"- Missing columns: {type_counts.get('Missing', 0)}",
        f"- Type errors: {type_counts.get('TypeError', 0)}",
        f"- Logic errors: {type_counts.get('LogicError', 0)}",
        f"- Calculation errors: {type_counts.get('CalcError', 0)}",
        "",
        "TOP 5 FILES WITH MOST ISSUES:",
    ]

    if top_files:
        for idx, (file_path, count) in enumerate(top_files.items(), 1):
            lines.append(f"{idx}. {file_path} - {count} issues")
    else:
        lines.append("1. None - 0 issues")

    lines.extend(["", "RECOMMENDED ACTIONS:"])
    for action in actions:
        lines.append(f"- {action}")

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    audit_df = run_audit()
    audit_df.to_csv(AUDIT_PATH, index=False)
    write_summary(audit_df)
    print(f"Audit complete. Issues: {len(audit_df)}")
    print(f"Saved: {AUDIT_PATH}")
    print(f"Saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
