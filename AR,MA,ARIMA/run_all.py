"""
run_all_models.py - Run all forecasting models in both modes
"""

import subprocess
import sys
import time
import argparse
import os


def log(msg: str):
    print(msg, flush=True)


def run_command(cmd: list, name: str) -> bool:
    """Run a command and return success status."""
    log(f"\n{'─'*50}")
    log(f"Running: {name}")
    log(f"{'─'*50}")
    
    t0 = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0
    
    if result.returncode == 0:
        log(f"✓ {name} completed in {elapsed:.2f}s")
        return True
    else:
        log(f"✗ {name} failed!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run All Forecasting Models")
    parser.add_argument("--data", required=True, help="Input CSV file")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon (default: 12)")
    parser.add_argument("--backtest", type=int, default=None, help="Backtest holdout periods")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    mode = "backtest" if args.backtest else "forecast"
    periods = args.backtest if args.backtest else args.horizon
    
    log(f"\n{'='*60}")
    log(f"RUNNING ALL MODELS - {mode.upper()} MODE")
    log(f"{'='*60}")
    log(f"Data: {args.data}")
    log(f"Mode: {mode}")
    log(f"Periods: {periods}")
    log(f"Output: {args.output_dir}/")
    
    results = {}
    
    models = [
        ("AR", "ar_forecast.py", []),
        ("MA", "ma_forecast.py", []),
        ("ARIMA", "arima_forecast.py", ["--auto"]),
    ]
    
    for name, script, extra_args in models:
        if args.backtest:
            cmd = [
                sys.executable, script,
                "--data", args.data,
                "--backtest", str(args.backtest),
                "--output", f"{args.output_dir}/{name.lower().replace(' ', '_')}_backtest.csv",
                "--verbose"
            ] + extra_args
        else:
            cmd = [
                sys.executable, script,
                "--data", args.data,
                "--horizon", str(args.horizon),
                "--output", f"{args.output_dir}/{name.lower().replace(' ', '_')}_forecast.csv",
                "--verbose"
            ] + extra_args
        
        results[name] = run_command(cmd, name)
    
    # Summary
    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}")
    
    for model, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        log(f"  {model:<20}: {status}")
    
    successful = sum(results.values())
    total = len(results)
    log(f"\nTotal: {successful}/{total} models completed")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()