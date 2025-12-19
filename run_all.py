#!/usr/bin/env python
"""
Polity Paper - Main Execution Script

This script runs all the analyses for the paper:
1. Generates model predictions
2. Creates plots
3. Runs statistical analysis

Usage:
    python run_all.py                    # Run everything
    python run_all.py --predictions      # Only run predictions
    python run_all.py --plots            # Only generate plots (requires predictions)
    python run_all.py --stats            # Only run statistics (requires predictions)
    python run_all.py --skip-bootstrap   # Skip bootstrap analysis (faster)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_path, description, extra_args=None):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}\n")
    
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    
    result = subprocess.run(cmd, cwd=script_path.parent)
    
    if result.returncode != 0:
        print(f"\nError: {description} failed with exit code {result.returncode}")
        return False
    
    print(f"\n{description} completed successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run all analyses for the Polity paper"
    )
    parser.add_argument(
        '--predictions', 
        action='store_true',
        help='Only run predictions'
    )
    parser.add_argument(
        '--plots', 
        action='store_true',
        help='Only generate plots (requires predictions to exist)'
    )
    parser.add_argument(
        '--stats', 
        action='store_true',
        help='Only run statistical analysis (requires predictions to exist)'
    )
    parser.add_argument(
        '--skip-bootstrap', 
        action='store_true',
        help='Skip bootstrap analysis in stats (much faster)'
    )
    
    args = parser.parse_args()
    
    # If no specific args, run everything
    run_all = not (args.predictions or args.plots or args.stats)
    
    # Get paths
    script_dir = Path(__file__).parent
    src_dir = script_dir / 'src'
    output_dir = script_dir / 'output'
    
    # Create output directories
    (output_dir / 'predictions').mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots').mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots' / 'si').mkdir(parents=True, exist_ok=True)
    
    print("Polity Paper Analysis Pipeline")
    print("="*60)
    print(f"Source directory: {src_dir}")
    print(f"Output directory: {output_dir}")
    
    success = True
    
    # Step 1: Run predictions
    if run_all or args.predictions:
        predictions_script = src_dir / 'run_predictions.py'
        if not run_script(predictions_script, "Model Predictions"):
            success = False
            if not (args.plots or args.stats):
                print("\nStopping due to prediction failure.")
                return 1
    
    # Check if predictions exist
    predictions_file = output_dir / 'predictions' / 'polity-short-4d-backfill.csv'
    
    # Step 2: Generate plots
    if run_all or args.plots:
        if not predictions_file.exists():
            print(f"\nError: Predictions file not found at {predictions_file}")
            print("Please run with --predictions first or without any flags.")
            return 1
        
        plots_script = src_dir / 'generate_plots.py'
        if not run_script(plots_script, "Plot Generation"):
            success = False
    
    # Step 3: Run statistical analysis
    if run_all or args.stats:
        if not predictions_file.exists():
            print(f"\nError: Predictions file not found at {predictions_file}")
            print("Please run with --predictions first or without any flags.")
            return 1
        
        stats_script = src_dir / 'generate_stats.py'
        extra_args = []
        if args.skip_bootstrap:
            extra_args.append('--skip-bootstrap')
        if not run_script(stats_script, "Statistical Analysis", extra_args):
            success = False
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("All analyses completed successfully!")
        print(f"\nOutputs saved to:")
        print(f"  Predictions: {output_dir / 'predictions'}")
        print(f"  Plots: {output_dir / 'plots'}")
        print(f"  SI Figures: {output_dir / 'plots' / 'si'}")
    else:
        print("Some analyses failed. Please check the output above.")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
