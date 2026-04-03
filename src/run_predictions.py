#!/usr/bin/env python
"""
4D models with BACKFILLING - Prediction Generation Script

This script generates predictions for GDP growth using various combinations
of predictors: Fitness, GDP, Polity, and Technological Fitness.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import itertools

from predictions import (
    keep_common_year_countries,
    take_logarithms,
    delete_below_fitness_threshold,
    delete_missing_data,
    interpolate_missing_values,
    make_predictions_at_all_dts_on_predictors,
    add_observable_values_to_df,
)
from imputation import apply_imputation


def main():
    parser = argparse.ArgumentParser(description='Generate predictions')
    parser.add_argument('--imputation', choices=['baseline', 'drop', 'country_min'],
                        default='baseline', help='TechFit NaN handling strategy')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory for predictions CSV')
    args = parser.parse_args()

    # Paths relative to this script location
    script_dir = Path(__file__).parent
    data_dir = script_dir / '../data'
    output_dir = args.output_dir or (script_dir / '../output/predictions')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define which data files to load
    renames = {
        'COMTRADE_reconciled_hmm/fitness.csv': 'fitness',
        'WDI/NY.GDP.PCAP.KD.csv': 'gdp',
    }

    selected_observables = renames.keys()

    # Load raw data
    raw_data_dict = dict()
    for obs in renames.keys():
        raw_data_dict[obs] = pd.read_csv(data_dir / obs, index_col=0)

    # Load polity data
    polity = pd.read_csv(data_dir / 'POLITY_V/polity2.csv').set_index('year').sort_index()

    # Z-score polity
    polity = (polity - np.nanmean(polity.values)) / np.nanstd(polity.values)

    raw_data_dict['polity'] = polity
    renames['polity'] = 'polity'

    # Process data
    keep_common_year_countries(raw_data_dict)
    observables = {key: raw_data_dict[key] for key in selected_observables}
    observables = {item: observables[key] for key, item in renames.items()}

    take_logarithms(observables, skip={'polity'})
    delete_below_fitness_threshold(observables, threshold=-6.)
    kill_report = delete_missing_data(observables)
    interpolate_missing_values(observables)

    print("Data shapes after initial processing:")
    print([(key, item.shape) for key, item in observables.items()])

    # Add tech fitness
    observables['tech_fitness'] = pd.read_csv(
        data_dir / 'PATSTAT_tech_fitness/tech_fitness_8dig.csv', index_col=0
    )
    keep_common_year_countries(observables)
    interpolate_missing_values(observables)

    print("\nData shapes after adding tech_fitness:")
    print([(key, item.shape) for key, item in observables.items()])

    # Apply TechFit imputation strategy
    print(f"\nImputation strategy: {args.imputation}")
    apply_imputation(observables, args.imputation)

    # Now delete still missing data
    kill_report = delete_missing_data(observables)

    print("\nData shapes after final processing:")
    print([(key, item.shape) for key, item in observables.items()])

    # Z-score tech fitness
    observables['tech_fitness'] = (
        observables['tech_fitness'] - np.nanmean(observables['tech_fitness'].values)
    ) / np.nanstd(observables['tech_fitness'].values)

    # Set up regression parameters
    bws = {
        'fitness': .3,
        'gdp': .3,
        'tech_fitness': .3,
        'polity': .09
    }
    
    all_models = sorted(
        itertools.chain.from_iterable(
            itertools.combinations(observables.keys(), J) 
            for J in range(1, len(bws) + 1)
        )
    )
    to_predict = 'gdp'

    print(f"\nRunning predictions for {len(all_models)} model combinations...")
    
    # Make predictions
    results = make_predictions_at_all_dts_on_predictors(all_models, bws, to_predict, observables)
    results = add_observable_values_to_df(results, observables)

    # Save results
    output_file = output_dir / 'polity-short-4d-backfill.csv'
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()

