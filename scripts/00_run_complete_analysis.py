"""
═══════════════════════════════════════════════════════════════════════
00_run_complete_analysis.py
Master Orchestrator — Runs All Analysis Scripts In Sequence
═══════════════════════════════════════════════════════════════════════
Usage:
  python scripts/00_run_complete_analysis.py            # Full analysis
  python scripts/00_run_complete_analysis.py --quick    # Skip visualizations
  python scripts/00_run_complete_analysis.py --report-only   # Reports 1 & 2 only
  python scripts/00_run_complete_analysis.py --forensic-only # Forensic only
  python scripts/00_run_complete_analysis.py --help     # Show usage
═══════════════════════════════════════════════════════════════════════
"""
import sys, os, time, subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOG_FILE = os.path.join(REPORTS_DIR, "analysis_log.txt")

SCRIPTS = [
    ("01_generate_performance_report.py",    "Performance Report",       "reports/performance"),
    ("02_generate_fixed_capital_analysis.py", "Fixed Capital Analysis",   "reports/fixed_capital"),
    ("03_generate_forensic_validation.py",    "Forensic Validation",      "reports/forensic_validation"),
    ("04_generate_visualizations.py",         "Visualization Suite",      "visualizations"),
]

Q3_SCRIPTS = [
    ("05_generate_q3_filter_analysis.py",     "Q3 Filter Analysis",       "reports/q3_filter"),
    ("06_generate_q3_filter_visualizations.py", "Q3 Visualizations",      "visualizations/q3_filter"),
]

TEMPORAL_SCRIPTS = [
    ("07_generate_temporal_optimization.py",  "Temporal Optimization",    "reports/temporal_optimization"),
    ("08_generate_temporal_visualizations.py", "Temporal Visualizations", "visualizations/temporal_optimization"),
]

PRECISION_SCRIPTS = [
    ("09_generate_precision_temporal_analysis.py", "Precision Temporal Analytics", "reports/precision_temporal"),
]

HELP_TEXT = """
═══════════════════════════════════════════════════════════════════════
  QUANTITATIVE ANALYSIS SUITE — USAGE
═══════════════════════════════════════════════════════════════════════

  python scripts/00_run_complete_analysis.py              Full analysis (all 4 scripts)
  python scripts/00_run_complete_analysis.py --quick      Skip visualizations (faster)
  python scripts/00_run_complete_analysis.py --report-only Reports 1 & 2 only
  python scripts/00_run_complete_analysis.py --forensic-only Forensic validation only
  python scripts/00_run_complete_analysis.py --help       Show this message

  Options:
    --include-q3                           Run standalone Q3 Filter Analysis
    --include-temporal                     Run Temporal Optimization Analysis
    --include-precision                    Run Ultra-Precision Temporal Engine

  Scripts:
    01 — Performance Report:       Core backtest metrics & statistics
    02 — Fixed Capital Analysis:   $100 capital with monthly withdrawals
    03 — Forensic Validation:      Monte Carlo, bootstrap, walk-forward, etc.
    04 — Visualization Suite:      26 professional charts (PNG)
  
  Q3 Filter Scripts (if --include-q3):
    05 — Q3 Filter Analysis:       Standalone Q3 filtered metrics & reports
    06 — Q3 Visualizations:        6 comparative charts evaluating Q3 filter

  Temporal Optimization Scripts (if --include-temporal):
    07 — Temporal Optimization:    Performance isolated by Hour/Day/Quarter
    08 — Temporal Visualizations:  7 heatmaps and distribution charts

  Precision Analytics Scripts (if --include-precision):
    09 — Precision Temporal:       Z-scores, rolling stability, and filter simulators

═══════════════════════════════════════════════════════════════════════
"""

def print_header():
    print("\n" + "═"*70)
    print("  INSTITUTIONAL QUANTITATIVE ANALYSIS SUITE")
    print("  Strategy Code: ALPHA-BTC-15M-v2.0")
    print(f"  Started: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}")
    print("═"*70)

def check_prereqs():
    data_file = os.path.join(BASE_DIR, "data", "BTCUSDT_15m_3_years.csv")
    if not os.path.exists(data_file):
        print(f"\n  ❌ ERROR: Data file not found: {data_file}")
        return False
    size_mb = os.path.getsize(data_file) / 1024 / 1024
    print(f"\n  ✅ Data file found: {size_mb:.1f} MB")
    return True

def run_script(script_name, label, output_dir):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    if not os.path.exists(script_path):
        return False, 0, f"Script not found: {script_path}"

    print(f"\n{'─'*70}")
    print(f"  ▶ Running: {label}")
    print(f"    Script:  {script_name}")
    print(f"{'─'*70}")

    t0 = time.time()
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, cwd=BASE_DIR, timeout=600,
            encoding='utf-8', errors='replace', env=env
        )
        elapsed = time.time() - t0
        stdout = result.stdout or ''
        stderr = result.stderr or ''

        # Log output
        with open(LOG_FILE, "a", encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Script: {script_name}\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"Duration: {elapsed:.1f}s\n")
            f.write(f"Exit Code: {result.returncode}\n")
            f.write(f"{'='*60}\n")
            if stdout: f.write(stdout)
            if stderr: f.write(f"\nSTDERR:\n{stderr}")

        if result.returncode == 0:
            print(f"\n  OK: {label} completed in {elapsed:.1f}s")
            # Print last few lines of output
            lines = stdout.strip().split('\n') if stdout.strip() else []
            for line in lines[-5:]:
                print(f"    {line}")
            return True, elapsed, ""
        else:
            print(f"\n  FAIL: {label} FAILED (exit code {result.returncode})")
            if stderr:
                for line in stderr.strip().split('\n')[-5:]:
                    print(f"    {line}")
            return False, elapsed, stderr[-500:] if stderr else "Unknown error"

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"\n  ❌ {label} TIMED OUT after {elapsed:.0f}s")
        return False, elapsed, "Timeout (600s limit)"
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  ❌ {label} ERROR: {e}")
        return False, elapsed, str(e)

def generate_master_summary(results, total_time):
    summary = f"""═══════════════════════════════════════════════════════════════════════
MASTER ANALYSIS SUMMARY
═══════════════════════════════════════════════════════════════════════
Generated: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}
Total Duration: {total_time:.1f} seconds

EXECUTION RESULTS:
{'─'*70}
"""
    for script, label, status, elapsed, error in results:
        sym = '✅' if status else '❌'
        summary += f"  {sym} {label:<35} {elapsed:>6.1f}s  {'OK' if status else 'FAILED'}\n"
        if error:
            summary += f"     Error: {error[:100]}\n"

    passed = sum(1 for _, _, s, _, _ in results if s)
    total = len(results)
    summary += f"""
{'─'*70}
  Passed: {passed}/{total}
  Status: {'ALL COMPLETE' if passed == total else 'PARTIAL FAILURE'}

OUTPUT FILES:
{'─'*70}
"""
    for dirpath, dirnames, filenames in os.walk(os.path.join(BASE_DIR, "reports")):
        for f in sorted(filenames):
            fp = os.path.join(dirpath, f)
            size = os.path.getsize(fp)
            rel = os.path.relpath(fp, BASE_DIR)
            summary += f"  {rel:<55} {size:>8,} bytes\n"

    viz_dir = os.path.join(BASE_DIR, "visualizations")
    if os.path.exists(viz_dir):
        pngs = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
        summary += f"\n  Visualizations: {len(pngs)} charts in visualizations/\n"

    q3_viz_dir = os.path.join(BASE_DIR, "visualizations", "q3_filter")
    if os.path.exists(q3_viz_dir):
        q3_pngs = [f for f in os.listdir(q3_viz_dir) if f.endswith('.png')]
        summary += f"  Q3 Visualizations: {len(q3_pngs)} charts in visualizations/q3_filter/\n"

    temporal_viz_dir = os.path.join(BASE_DIR, "visualizations", "temporal_optimization")
    if os.path.exists(temporal_viz_dir):
        t_pngs = [f for f in os.listdir(temporal_viz_dir) if f.endswith('.png')]
        summary += f"  Temporal Visualizations: {len(t_pngs)} charts in visualizations/temporal_optimization/\n"

    precision_dir = os.path.join(BASE_DIR, "reports", "precision_temporal")
    if os.path.exists(precision_dir):
        p_csvs = [f for f in os.listdir(precision_dir) if f.endswith('.csv')]
        summary += f"  Precision Analytics:   {len(p_csvs)} matrices in reports/precision_temporal/\n"

    summary += f"""
═══════════════════════════════════════════════════════════════════════
  Classification: CONFIDENTIAL
  Strategy Code: ALPHA-BTC-15M-v2.0
═══════════════════════════════════════════════════════════════════════
"""

    summary_path = os.path.join(REPORTS_DIR, "MASTER_SUMMARY.txt")
    with open(summary_path, "w", encoding='utf-8') as f:
        f.write(summary)
    return summary_path

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    args = sys.argv[1:]

    if '--help' in args or '-h' in args:
        print(HELP_TEXT)
        return

    # Determine which scripts to run
    if '--forensic-only' in args:
        scripts_to_run = [SCRIPTS[2]]  # Only forensic
    elif '--report-only' in args:
        scripts_to_run = SCRIPTS[:2]   # Reports 1 & 2
    elif '--quick' in args:
        scripts_to_run = SCRIPTS[:3]   # Skip visualizations
    else:
        scripts_to_run = SCRIPTS[:]    # All

    if '--include-q3' in args:
        scripts_to_run.extend(Q3_SCRIPTS)
        
    if '--include-temporal' in args:
        scripts_to_run.extend(TEMPORAL_SCRIPTS)
        
    if '--include-precision' in args:
        scripts_to_run.extend(PRECISION_SCRIPTS)

    print_header()

    # Create directories
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Initialize log
    with open(LOG_FILE, "w", encoding='utf-8') as f:
        f.write(f"Analysis Log — {datetime.now().isoformat()}\n{'='*60}\n")

    # Check prerequisites
    print("\n  Checking prerequisites...")
    if not check_prereqs():
        print("\n  ❌ Prerequisites check failed. Aborting.")
        sys.exit(1)

    print(f"\n  📋 Running {len(scripts_to_run)} scripts...")
    for i, (name, label, _) in enumerate(scripts_to_run, 1):
        print(f"    {i}. {label}")

    # Execute scripts
    total_start = time.time()
    results = []

    for i, (script_name, label, output_dir) in enumerate(scripts_to_run, 1):
        remaining = len(scripts_to_run) - i
        if remaining > 0:
            print(f"\n  ⏳ {remaining} script(s) remaining after this...")
        success, elapsed, error = run_script(script_name, label, output_dir)
        results.append((script_name, label, success, elapsed, error))

        if not success:
            print(f"\n  ⚠️  {label} failed. Continuing with remaining scripts...")

    total_time = time.time() - total_start

    # Generate master summary
    summary_path = generate_master_summary(results, total_time)

    # Final report
    passed = sum(1 for _, _, s, _, _ in results if s)
    total = len(results)

    print(f"\n{'═'*70}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'═'*70}")
    print(f"  Scripts Run:    {total}")
    print(f"  Passed:         {passed}/{total}")
    print(f"  Total Time:     {total_time:.1f} seconds")
    print(f"  Master Summary: {summary_path}")
    print(f"  Full Log:       {LOG_FILE}")
    print(f"{'═'*70}")

    if passed < total:
        print(f"\n  ⚠️  {total - passed} script(s) failed. Check log for details.")
        sys.exit(1)
    else:
        print(f"\n  ✅ All scripts completed successfully!")

if __name__ == "__main__":
    main()
