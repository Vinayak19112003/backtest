import os
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")

scripts_to_run = [
    "15_validate_drawdown_calculations.py",
    "01_generate_performance_report.py",
    "02_generate_fixed_capital_analysis.py",
    "03_generate_forensic_validation.py",
    "05_generate_q3_filter_analysis.py",
    "12_generate_exhaustive_filter_analysis.py",
    "13_generate_institutional_reports_all_filters.py"
]

def run_script(script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    print(f"\n{'='*70}\nExecuting: {script_name}\n{'='*70}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    result = subprocess.run(["python", script_path], capture_output=False, env=env)
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(f"\n[!] Error executing {script_name}. Return code: {result.returncode}")
        return False
    else:
        print(f"\n[+] Successfully executed {script_name} in {elapsed:.1f}s")
        return True

def main():
    print("Starting Drawdown Calculation Fix Orchestrator\n")
    print("This will regenerate all reports using the corrected Drawdown anchoring logic.")
    
    total_t0 = time.time()
    
    for script in scripts_to_run:
        success = run_script(script)
        if not success:
            print("\nOrchestrator aborted due to errors.")
            return

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"All {len(scripts_to_run)} scripts completed successfully in {total_elapsed:.1f}s.")
    print("Drawdown calculations refactored and all reports regenerated.")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
