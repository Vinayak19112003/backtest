"""
15_validate_drawdown_calculations.py

Unit tests to validate the absolute Drawdown calculation method where Drawdown 
is anchored strictly to INITIAL_CAPITAL instead of Peak Equity.
"""

import pandas as pd
import numpy as np

def calculate_drawdown_correct(cumulative_pnl_series, initial_capital=100.0):
    """
    Calculate drawdown using traditional peak-to-valley method,
    but express as percentage of INITIAL CAPITAL, not peak.
    """
    if len(cumulative_pnl_series) == 0:
        return {
            'dd_dollars_series': pd.Series(dtype=float),
            'dd_pct_series': pd.Series(dtype=float),
            'max_dd_dollars': 0.0,
            'max_dd_pct': 0.0,
            'max_dd_date': None,
            'peak_before_max_dd': initial_capital,
            'valley_at_max_dd': initial_capital,
            'equity_series': pd.Series(dtype=float),
            'peak_series': pd.Series(dtype=float)
        }
        
    # Calculate equity curve
    equity_series = initial_capital + cumulative_pnl_series
    
    # Calculate running peak (high water mark)
    peak_series = equity_series.expanding().max()
    
    # Drawdown in DOLLARS (peak - current)
    dd_dollars_series = peak_series - equity_series
    
    # Drawdown as PERCENTAGE OF INITIAL CAPITAL
    dd_pct_series = (dd_dollars_series / initial_capital) * 100
    
    # Find maximum drawdown
    max_dd_idx = dd_dollars_series.idxmax()
    max_dd_dollars = dd_dollars_series[max_dd_idx]
    max_dd_pct = dd_pct_series[max_dd_idx]
    
    # Find peak before max DD and valley at max DD
    peak_before_max_dd = peak_series[max_dd_idx]
    valley_at_max_dd = equity_series[max_dd_idx]
    
    return {
        'dd_dollars_series': dd_dollars_series,
        'dd_pct_series': dd_pct_series,
        'max_dd_dollars': max_dd_dollars,
        'max_dd_pct': max_dd_pct,
        'max_dd_date': max_dd_idx,
        'peak_before_max_dd': peak_before_max_dd,
        'valley_at_max_dd': valley_at_max_dd,
        'equity_series': equity_series,
        'peak_series': peak_series
    }

def run_tests():
    print("Running Drawdown Calculation Unit Tests...\n")
    
    # Test case: $100 -> $200 -> $170 -> $205
    # Pnl series: $50 -> $100 -> $70 -> $105
    test1_pnl = pd.Series([50.0, 100.0, 70.0, 105.0], index=['d1', 'd2', 'd3', 'd4'])
    res1 = calculate_drawdown_correct(test1_pnl, initial_capital=100)
    
    assert res1['peak_before_max_dd'] == 200  # Peak at index d2 (100 + 100)
    assert res1['valley_at_max_dd'] == 170    # Valley at index d3 (100 + 70)
    assert res1['max_dd_dollars'] == 30       # $200 - $170 = $30
    assert res1['max_dd_pct'] == 30.0         # $30 / $100 = 30%
    print("[PASS] Test 1: $100 -> $200 -> $170 -> $205 (Max DD: 30.0%)")

    # Test case: Pure Profit, no retracement
    test2_pnl = pd.Series([10.0, 20.0, 30.0, 40.0], index=['d1', 'd2', 'd3', 'd4'])
    res2 = calculate_drawdown_correct(test2_pnl, initial_capital=100)
    assert res2['max_dd_pct'] == 0.0
    print("[PASS] Test 2: Pure Profit (Max DD: 0.0%)")

    # Test case: Deep loss below initial capital 
    # $100 -> $110 -> $50 -> $150
    # PnL: $10 -> -$50 -> $50
    test3_pnl = pd.Series([10.0, -50.0, 50.0], index=['d1', 'd2', 'd3'])
    res3 = calculate_drawdown_correct(test3_pnl, initial_capital=100)
    assert res3['max_dd_pct'] == 60.0
    print("[PASS] Test 3: Drop below initial capital, $110 peak to $50 valley (Max DD: 60%)")

    print("\n[SUCCESS] All Drawdown Validation Tests Passed Successfully!")

if __name__ == "__main__":
    run_tests()
