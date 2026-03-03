import os
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

class ResultsManager:
    """Manages organized storage of backtest results"""
    
    def __init__(self, base_dir='results'):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def create_strategy_folder(self, strategy_name):
        """
        Create folder for strategy with parameters in name
        Example: results/rsi_43_58_bb_1.2/
        """
        folder_path = os.path.join(self.base_dir, strategy_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def save_all_results(self, strategy_name, trades_df, summary, 
                        train_metrics, val_metrics, test_metrics, params):
        """
        Save all results to strategy folder
        
        Args:
            strategy_name: e.g., "rsi_43_58_bb_1.2"
            trades_df: All trades DataFrame
            summary: Overall performance dict
            train/val/test_metrics: Split performance dicts
            params: Strategy parameters dict
        """
        folder = self.create_strategy_folder(strategy_name)
        
        print(f"\n[SAVE] Saving results to: {folder}/")
        
        # 1. Trades CSV
        trades_file = os.path.join(folder, 'trades.csv')
        trades_df.to_csv(trades_file, index=False)
        print(f"  OK trades.csv ({len(trades_df)} trades)")
        
        # 2. Performance Summary JSON
        performance_data = {
            'strategy': strategy_name,
            'parameters': params,
            'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'overall': summary,
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
        
        summary_file = os.path.join(folder, 'performance_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(performance_data, f, indent=2, default=str)
        print(f"  OK performance_summary.json")
        
        # 3. Equity Curve PNG
        if len(trades_df) > 0:
            plt.figure(figsize=(14, 7))
            
            # Plot equity curve
            plt.subplot(2, 1, 1)
            plt.plot(trades_df['timestamp'], trades_df['capital_after'], linewidth=2)
            plt.title(f'{strategy_name} - Equity Curve', fontsize=14, fontweight='bold')
            plt.ylabel('Capital ($)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Plot drawdown
            plt.subplot(2, 1, 2)
            cumulative = trades_df['pnl'].cumsum()
            running_max = cumulative.cummax()
            drawdown = cumulative - running_max
            plt.fill_between(trades_df['timestamp'], drawdown, 0, alpha=0.3, color='red')
            plt.plot(trades_df['timestamp'], drawdown, color='darkred', linewidth=1)
            plt.title('Drawdown', fontsize=12)
            plt.ylabel('Drawdown ($)', fontsize=12)
            plt.xlabel('Date', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            equity_file = os.path.join(folder, 'equity_curve.png')
            plt.savefig(equity_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  OK equity_curve.png")
        
        # 4. Regime Analysis CSV
        if len(trades_df) > 0 and 'regime_vol' in trades_df.columns:
            regime_data = []
            
            for vol_regime in ['Low', 'Medium', 'High']:
                for trend_regime in ['Weak', 'Strong']:
                    subset = trades_df[
                        (trades_df['regime_vol'] == vol_regime) & 
                        (trades_df['regime_trend'] == trend_regime)
                    ]
                    
                    if len(subset) > 0:
                        loss_trades = subset[~subset['win']]
                        avg_risk = abs(loss_trades['pnl'].mean()) if len(loss_trades) > 0 else 1.0
                        
                        regime_data.append({
                            'volatility': vol_regime,
                            'trend': trend_regime,
                            'trades': len(subset),
                            'win_rate': subset['win'].mean(),
                            'total_pnl': subset['pnl'].sum(),
                            'avg_pnl': subset['pnl'].mean(),
                            'sharpe': (subset['pnl'] / avg_risk).mean() / (subset['pnl'] / avg_risk).std() if len(subset) > 1 else 0
                        })
            
            regime_file = os.path.join(folder, 'regime_analysis.csv')
            pd.DataFrame(regime_data).to_csv(regime_file, index=False)
            print(f"  OK regime_analysis.csv")
        
        # 5. Text Report
        report_file = os.path.join(folder, 'backtest_report.txt')
        with open(report_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"BACKTEST REPORT: {strategy_name}\n")
            f.write("="*70 + "\n\n")
            
            f.write("PARAMETERS:\n")
            for key, val in params.items():
                f.write(f"  {key}: {val}\n")
            
            f.write("\nOVERALL PERFORMANCE:\n")
            f.write(f"  Total Trades:   {summary.get('total_trades', 0)}\n")
            f.write(f"  Win Rate:       {summary.get('win_rate', 0):.2%}\n")
            f.write(f"  Total PnL:      ${summary.get('total_pnl', 0):.2f}\n")
            f.write(f"  Total Return:   {summary.get('total_return_pct', 0):.2f}%\n")
            f.write(f"  Sharpe Ratio:   {summary.get('sharpe_ratio', 0):.2f}\n")
            f.write(f"  Max Drawdown:   ${summary.get('max_drawdown', 0):.2f} ({summary.get('max_drawdown_pct', 0):.1f}%)\n")
            f.write(f"  Max DD Days:    {summary.get('max_drawdown_days', 0)} days\n")
            f.write(f"  Profit Factor:  {summary.get('profit_factor', 0):.2f}\n")
            f.write(f"  P-value:        {summary.get('p_value', 0):.4f}\n")
            
            f.write("\n" + "-"*70 + "\n")
            f.write("PERIOD BREAKDOWN:\n\n")
            
            for period_name, metrics in [('TRAIN', train_metrics), 
                                         ('VALIDATION', val_metrics), 
                                         ('TEST', test_metrics)]:
                f.write(f"{period_name}:\n")
                if metrics:
                    f.write(f"  Trades:      {metrics.get('total_trades', 0)}\n")
                    f.write(f"  Win Rate:    {metrics.get('win_rate', 0):.2%}\n")
                    f.write(f"  Total PnL:   ${metrics.get('total_pnl', 0):.2f}\n")
                    f.write(f"  Sharpe:      {metrics.get('sharpe_ratio', 0):.2f}\n")
                    f.write(f"  Max DD:      ${metrics.get('max_drawdown', 0):.2f}\n")
                    f.write(f"  Max DD Days: {metrics.get('max_drawdown_days', 0)} days\n")
                    f.write(f"  P-value:     {metrics.get('p_value', 0):.4f}\n\n")
                else:
                    f.write("  No trades\n\n")
            
            # Verdict
            f.write("-"*70 + "\n")
            f.write("VERDICT:\n")
            
            test_win_rate = test_metrics.get('win_rate', 0) if test_metrics else 0
            test_p_value = test_metrics.get('p_value', 1) if test_metrics else 1
            test_sharpe = test_metrics.get('sharpe_ratio', 0) if test_metrics else 0
            test_trades = test_metrics.get('total_trades', 0) if test_metrics else 0
            
            f.write(f"  {'Y' if test_win_rate > 0.52 else 'N'} Win Rate > 52% (breakeven)\n")
            f.write(f"  {'Y' if test_p_value < 0.05 else 'N'} Statistically significant (p < 0.05)\n")
            f.write(f"  {'Y' if test_sharpe > 1.0 else 'N'} Sharpe Ratio > 1.0\n")
            f.write(f"  {'Y' if test_trades >= 100 else 'N'} Sufficient sample size (>100 trades)\n")
        
        print(f"  OK backtest_report.txt")
        print(f"\nDONE All results saved to: {folder}/\n")
        
        return folder
