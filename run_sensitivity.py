#!/usr/bin/env python
"""Run the full pipeline for each TechFit imputation strategy and generate comparisons."""

import argparse
import subprocess
import sys
from pathlib import Path

STRATEGIES = ['baseline', 'drop', 'country_min']


def run_pipeline(strategy, recompute_dir, script_dir, skip_bootstrap=False):
    """Run run_all.py for one imputation strategy."""
    output_dir = recompute_dir / strategy
    cmd = [
        sys.executable, str(script_dir / 'run_all.py'),
        '--imputation', strategy,
        '--output-dir', str(output_dir),
    ]
    if skip_bootstrap:
        cmd.append('--skip-bootstrap')

    print(f"\n{'='*60}")
    print(f"Running pipeline: strategy={strategy}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    assert result.returncode == 0, f"Pipeline failed for strategy '{strategy}'"


def run_comparison(base_output_dir, src_dir):
    """Run generate_comparison.py."""
    cmd = [
        sys.executable, str(src_dir / 'generate_comparison.py'),
        '--base-output-dir', str(base_output_dir),
    ]

    print(f"\n{'='*60}")
    print("Generating comparison artefacts")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(src_dir))
    assert result.returncode == 0, "Comparison generation failed"


def main():
    parser = argparse.ArgumentParser(
        description='Run sensitivity analysis for TechFit imputation strategies')
    parser.add_argument('--skip-bootstrap', action='store_true',
                        help='Skip bootstrap analysis (much faster)')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Base output directory (default: output/)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    src_dir = script_dir / 'src'
    base_output_dir = args.output_dir or (script_dir / 'output')
    comparison_dir = base_output_dir / 'sensitivity_comparison'
    recompute_dir = comparison_dir / 'recompute_plots_with_other_strategies'

    for strategy in STRATEGIES:
        run_pipeline(strategy, recompute_dir, script_dir, args.skip_bootstrap)

    run_comparison(base_output_dir, src_dir)

    print(f"\n{'='*60}")
    print("Sensitivity analysis complete!")
    print(f"Results: {comparison_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    sys.exit(main() or 0)
