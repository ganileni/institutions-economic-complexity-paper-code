"""TechFit imputation strategies for sensitivity analysis."""

import numpy as np


def apply_imputation(observables, strategy):
    """Apply an imputation strategy to tech_fitness NaN values in-place.

    Args:
        observables: dict of DataFrames (must include 'tech_fitness')
        strategy: one of 'baseline', 'drop', 'country_min'
    """
    tf = observables['tech_fitness']

    if strategy == 'baseline':
        minimum_tf = np.nanmin(tf.values)
        observables['tech_fitness'] = tf.fillna(minimum_tf)

    elif strategy == 'drop':
        pass  # leave NaN; delete_missing_data will propagate

    elif strategy == 'country_min':
        observables['tech_fitness'] = tf.bfill()

    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")
