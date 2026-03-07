"""
Phase 2: Clean CSV artifacts based on the data quality audit.
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
AUDIT_PATH = REPORTS_DIR / "data_quality_audit.csv"
LOG_PATH = REPORTS_DIR / "cleaning_log.txt"

REQUIRED_BY_FILE = {
    "reports/exhaustive_filter_analysis/01_basic_metrics.csv": {
        "strategy_name",
        "total_trades",
        "win_count",
        "loss_count",
        "win_rate",
        "total_pnl",
        "direction_yes_trades",
        "direction_no_trades",
    },
    "reports/exhaustive_filter_analysis/02_risk_metrics.csv": {
        "strategy_name",
        "sharpe_ratio",
        "sortino_ratio",
        "profit_factor",
        "max_drawdown_pct",
    },
    "reports/exhaustive_filter_analysis/07_drawdown_analysis.csv": {
        "strategy_name",
        "max_drawdown_duration_days",
        "max_drawdown_depth_dollars",
        "max_drawdown_depth_pct",
    },
    "reports/exhaustive_filter_analysis/10_comparative_matrix.csv": {
        "strategy_name",
        "total_trades",
        "win_rate",
        "total_pnl",
    },
}

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

PERCENT_COLUMNS = {
    "win_rate",
    "loss_rate",
    "direction_yes_pct",
    "direction_no_pct",
    "max_drawdown_pct",
    "max_drawdown_depth_pct",
    "trades_removed_pct",
    "mc_probability_profitable",
    "mc_risk_of_ruin_pct",
    "year_2023_win_rate",
    "year_2024_win_rate",
    "year_2025_win_rate",
    "year_2026_win_rate",
    "quarterly_win_rate_q1",
    "quarterly_win_rate_q2",
    "quarterly_win_rate_q3",
    "quarterly_win_rate_q4",
}

MONEY_HINTS = (
    "pnl",
    "profit",
    "loss",
    "fees",
    "drawdown_dollars",
    "depth_dollars",
    "peak_before_max_dd",
    "valley_at_max_dd",
)


@dataclass
class FixRecord:
    file_path: str
    issue: str
    fix_applied: str
    before: str
    after: str
    status: str


def _rel(path: Path) -> str:
    return path.relative_to(BASE_DIR).as_posix()


def _to_str(value: Any) -> str:
    text = str(value)
    return text[:200]


def _read_audit() -> pd.DataFrame:
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(f"Missing audit file: {AUDIT_PATH}")
    df = pd.read_csv(AUDIT_PATH)
    if df.empty:
        return df
    needed = {
        "file_path",
        "issue_type",
        "column_name",
        "affected_rows",
        "severity",
        "example_value",
        "suggested_fix",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Audit file missing required columns: {sorted(missing)}")
    return df


def _replace_inf_nan(df: pd.DataFrame, rel_path: str, records: list[FixRecord]) -> pd.DataFrame:
    out = df.copy()
    numeric = out.select_dtypes(include=[np.number]).columns
    for col in numeric:
        series = out[col]
        inf_mask = np.isinf(series.to_numpy())
        if inf_mask.any():
            before = series[inf_mask].iloc[0]
            out.loc[inf_mask, col] = 0.0
            records.append(
                FixRecord(
                    rel_path,
                    f"Infinite values in {col}",
                    "Replaced +/-inf with 0.0 for denominator-zero guard.",
                    _to_str(before),
                    "0.0",
                    "Fixed",
                )
            )

        nan_mask = out[col].isna()
        if nan_mask.any():
            before = out.loc[nan_mask, col].iloc[0]
            out.loc[nan_mask, col] = 0.0
            records.append(
                FixRecord(
                    rel_path,
                    f"NaN values in {col}",
                    "Filled numeric NaN with 0.0 pending recalculation pipeline.",
                    _to_str(before),
                    "0.0",
                    "Fixed",
                )
            )

    object_cols = out.select_dtypes(exclude=[np.number]).columns
    for col in object_cols:
        nan_mask = out[col].isna()
        if nan_mask.any():
            before = out.loc[nan_mask, col].iloc[0]
            out.loc[nan_mask, col] = ""
            records.append(
                FixRecord(
                    rel_path,
                    f"NaN values in {col}",
                    "Filled text NaN with empty string to preserve CSV consistency.",
                    _to_str(before),
                    "",
                    "Fixed",
                )
            )

    return out


def _apply_calculation_fixes(df: pd.DataFrame, rel_path: str, records: list[FixRecord]) -> pd.DataFrame:
    out = df.copy()

    if {"win_count", "loss_count", "total_trades"}.issubset(out.columns):
        win = pd.to_numeric(out["win_count"], errors="coerce").fillna(0)
        loss = pd.to_numeric(out["loss_count"], errors="coerce").fillna(0)
        total = pd.to_numeric(out["total_trades"], errors="coerce").fillna(0)
        corrected_total = win + loss
        bad = (corrected_total - total).abs() > 1e-6
        if bad.any():
            idx = bad[bad].index[0]
            before = total.loc[idx]
            out.loc[bad, "total_trades"] = corrected_total[bad]
            records.append(
                FixRecord(
                    rel_path,
                    "total_trades mismatch with win_count + loss_count",
                    "Set total_trades = win_count + loss_count.",
                    _to_str(before),
                    _to_str(corrected_total.loc[idx]),
                    "Fixed",
                )
            )

    if {"direction_yes_trades", "direction_no_trades", "total_trades"}.issubset(out.columns):
        yes = pd.to_numeric(out["direction_yes_trades"], errors="coerce").fillna(0)
        no = pd.to_numeric(out["direction_no_trades"], errors="coerce").fillna(0)
        total = pd.to_numeric(out["total_trades"], errors="coerce").fillna(0)
        bad = ((yes + no) - total).abs() > 1e-6
        if bad.any():
            idx = bad[bad].index[0]
            before_yes = yes.loc[idx]
            before_no = no.loc[idx]
            delta = total - (yes + no)
            adjust_yes = yes + delta
            out.loc[bad, "direction_yes_trades"] = adjust_yes[bad]
            records.append(
                FixRecord(
                    rel_path,
                    "Direction count mismatch with total_trades",
                    "Adjusted direction_yes_trades so YES + NO equals total_trades.",
                    f"yes={before_yes}, no={before_no}, total={total.loc[idx]}",
                    f"yes={adjust_yes.loc[idx]}, no={no.loc[idx]}, total={total.loc[idx]}",
                    "Fixed",
                )
            )

    if {"total_profit", "total_loss", "total_pnl"}.issubset(out.columns):
        tp = pd.to_numeric(out["total_profit"], errors="coerce").fillna(0)
        tl = pd.to_numeric(out["total_loss"], errors="coerce").fillna(0)
        pnl = pd.to_numeric(out["total_pnl"], errors="coerce").fillna(0)
        expected = tp - tl
        bad = (expected - pnl).abs() > 1e-4
        if bad.any():
            idx = bad[bad].index[0]
            before = pnl.loc[idx]
            out.loc[bad, "total_pnl"] = expected[bad]
            records.append(
                FixRecord(
                    rel_path,
                    "total_pnl mismatch with total_profit - total_loss",
                    "Set total_pnl = total_profit - total_loss.",
                    _to_str(before),
                    _to_str(expected.loc[idx]),
                    "Fixed",
                )
            )

    if {"win_count", "loss_count", "total_trades", "win_rate", "loss_rate"}.issubset(out.columns):
        total = pd.to_numeric(out["total_trades"], errors="coerce").replace(0, np.nan)
        win = pd.to_numeric(out["win_count"], errors="coerce").fillna(0)
        loss = pd.to_numeric(out["loss_count"], errors="coerce").fillna(0)
        out["win_rate"] = (win / total * 100).fillna(0)
        out["loss_rate"] = (loss / total * 100).fillna(0)

    if {"direction_yes_trades", "direction_no_trades", "total_trades"}.issubset(out.columns):
        total = pd.to_numeric(out["total_trades"], errors="coerce").replace(0, np.nan)
        yes = pd.to_numeric(out["direction_yes_trades"], errors="coerce").fillna(0)
        no = pd.to_numeric(out["direction_no_trades"], errors="coerce").fillna(0)
        if "direction_yes_pct" in out.columns:
            out["direction_yes_pct"] = (yes / total * 100).fillna(0)
        if "direction_no_pct" in out.columns:
            out["direction_no_pct"] = (no / total * 100).fillna(0)

    if {"win", "pnl"}.issubset(out.columns):
        pnl = pd.to_numeric(out["pnl"], errors="coerce")
        win_as_bool = out["win"].astype(str).str.lower().map({"true": True, "false": False})
        mismatch = ((win_as_bool == True) & (pnl <= 0)) | ((win_as_bool == False) & (pnl >= 0))
        mismatch = mismatch.fillna(False)
        if mismatch.any():
            before = out.loc[mismatch, "win"].iloc[0]
            out.loc[mismatch, "win"] = (pnl[mismatch] > 0)
            records.append(
                FixRecord(
                    rel_path,
                    "win flag inconsistent with pnl sign",
                    "Set win = (pnl > 0).",
                    _to_str(before),
                    _to_str(out.loc[mismatch, "win"].iloc[0]),
                    "Fixed",
                )
            )

    if "signal" in out.columns:
        signal = out["signal"].astype(str).str.upper().str.strip()
        bad = ~signal.isin(["YES", "NO"])
        if bad.any():
            records.append(
                FixRecord(
                    rel_path,
                    "Invalid signal values",
                    "Kept original value for manual review.",
                    _to_str(out.loc[bad, "signal"].iloc[0]),
                    _to_str(out.loc[bad, "signal"].iloc[0]),
                    "Cannot Fix",
                )
            )
        out["signal"] = signal

    return out


def _normalize_types(df: pd.DataFrame, rel_path: str, records: list[FixRecord]) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if col in COUNT_COLUMNS:
            numeric = pd.to_numeric(out[col], errors="coerce")
            out[col] = numeric.round(0).fillna(0).astype(int)

    for col in out.columns:
        if col.endswith("_date") or col == "timestamp":
            parsed = pd.to_datetime(out[col], errors="coerce")
            if col.endswith("_date"):
                out[col] = parsed.dt.strftime("%Y-%m-%d").fillna("N/A")
            else:
                # Keep timestamp granularity while ensuring stable parse.
                out[col] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("N/A")

    for col in out.columns:
        if col in PERCENT_COLUMNS:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).round(2)
        elif any(hint in col for hint in MONEY_HINTS):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).round(2)

    records.append(
        FixRecord(
            rel_path,
            "Type normalization",
            "Applied standard formatting: integer counts, YYYY-MM-DD dates, 2-decimal percentages and currency fields.",
            "mixed",
            "normalized",
            "Fixed",
        )
    )
    return out


def _validate_cleaned(df: pd.DataFrame, rel_path: str) -> tuple[bool, list[str]]:
    errors: list[str] = []

    required = REQUIRED_BY_FILE.get(rel_path, set())
    if rel_path.startswith("reports/exhaustive_filter_analysis/12_trades_"):
        required = {"trade_id", "timestamp", "signal", "win", "pnl", "fees"}

    missing = sorted(required - set(df.columns))
    if missing:
        errors.append(f"Missing columns: {missing}")

    if {"win_count", "loss_count", "total_trades"}.issubset(df.columns):
        bad = (
            pd.to_numeric(df["win_count"], errors="coerce").fillna(0)
            + pd.to_numeric(df["loss_count"], errors="coerce").fillna(0)
            - pd.to_numeric(df["total_trades"], errors="coerce").fillna(0)
        ).abs() > 1e-6
        if bad.any():
            errors.append("total_trades != win_count + loss_count")

    if {"direction_yes_trades", "direction_no_trades", "total_trades"}.issubset(df.columns):
        bad = (
            pd.to_numeric(df["direction_yes_trades"], errors="coerce").fillna(0)
            + pd.to_numeric(df["direction_no_trades"], errors="coerce").fillna(0)
            - pd.to_numeric(df["total_trades"], errors="coerce").fillna(0)
        ).abs() > 1e-6
        if bad.any():
            errors.append("total_trades != direction_yes_trades + direction_no_trades")

    for col in ["win_rate", "loss_rate", "direction_yes_pct", "direction_no_pct"]:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = (vals < 0) | (vals > 100)
        bad = bad.fillna(False)
        if bad.any():
            errors.append(f"{col} out of [0, 100]")

    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        if np.isinf(numeric.to_numpy()).any():
            errors.append("Infinite numeric values remain")
    if df.isna().sum().sum() > 0:
        errors.append("NaN values remain")

    return len(errors) == 0, errors


def _write_log(records: list[FixRecord], updated_files: list[str], partial_files: list[str], cannot_files: list[str]) -> None:
    generated = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    fixed_count = sum(1 for r in records if r.status == "Fixed")
    partial_count = sum(1 for r in records if r.status == "Partial")
    cannot_count = sum(1 for r in records if r.status == "Cannot Fix")

    lines = [
        "DATA CLEANING LOG",
        f"Generated: {generated}",
        "Backtest: Polymarket 15m BTC UP/DOWN prediction",
        "",
        f"ISSUES FIXED: {fixed_count}",
        "",
        "DETAILED FIXES:",
    ]

    if records:
        for rec in records:
            lines.extend(
                [
                    f"File: {rec.file_path}",
                    f"  Issue: {rec.issue}",
                    f"  Fix Applied: {rec.fix_applied}",
                    f"  Before: {rec.before}",
                    f"  After: {rec.after}",
                    f"  Status: {rec.status}",
                    "",
                ]
            )
    else:
        lines.append("No fixes were required.")
        lines.append("")

    lines.extend(
        [
            "SUMMARY:",
            f"- Successfully fixed: {fixed_count}",
            f"- Partially fixed: {partial_count}",
            f"- Cannot fix: {cannot_count}",
            "",
            "FILES UPDATED:",
        ]
    )

    for path in updated_files:
        lines.append(f"- {path}")
    if not updated_files:
        lines.append("- None")

    if partial_files:
        lines.append("")
        lines.append("FILES WITH PARTIAL VALIDATION:")
        for path in partial_files:
            lines.append(f"- {path}")

    if cannot_files:
        lines.append("")
        lines.append("FILES WITH CANNOT FIX ITEMS:")
        for path in cannot_files:
            lines.append(f"- {path}")

    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    audit_df = _read_audit()
    target_files = sorted((BASE_DIR / p).resolve() for p in audit_df["file_path"].dropna().unique()) if not audit_df.empty else []

    records: list[FixRecord] = []
    updated_files: list[str] = []
    partial_files: list[str] = []
    cannot_files: list[str] = []

    for file_path in target_files:
        if not file_path.exists() or file_path.suffix.lower() != ".csv":
            continue

        rel_path = _rel(file_path)
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:  # noqa: BLE001
            records.append(
                FixRecord(
                    rel_path,
                    "Unreadable CSV",
                    "Skipped file due parse error.",
                    str(exc),
                    str(exc),
                    "Cannot Fix",
                )
            )
            cannot_files.append(rel_path)
            continue

        cleaned = _replace_inf_nan(df, rel_path, records)
        cleaned = _apply_calculation_fixes(cleaned, rel_path, records)
        cleaned = _normalize_types(cleaned, rel_path, records)

        cleaned_path = file_path.with_name(f"{file_path.stem}_cleaned.csv")
        cleaned.to_csv(cleaned_path, index=False)

        is_valid, validation_errors = _validate_cleaned(cleaned, rel_path)
        if is_valid:
            cleaned.to_csv(file_path, index=False)
            try:
                cleaned_path.unlink(missing_ok=True)
            except OSError:
                pass
            updated_files.append(rel_path)
        else:
            try:
                cleaned_path.unlink(missing_ok=True)
            except OSError:
                pass
            partial_files.append(rel_path)
            records.append(
                FixRecord(
                    rel_path,
                    "Validation errors after cleaning",
                    "Kept original file for safety.",
                    "; ".join(validation_errors),
                    "; ".join(validation_errors),
                    "Partial",
                )
            )

    _write_log(records, updated_files, partial_files, cannot_files)
    print(f"Cleaning complete. Updated files: {len(updated_files)}")
    print(f"Log saved: {LOG_PATH}")


if __name__ == "__main__":
    main()



