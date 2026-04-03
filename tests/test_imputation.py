"""Tests for TechFit imputation strategies."""

import numpy as np
import pandas as pd
import pytest

from imputation import apply_imputation


def _make_observables():
    """Create a small observables dict with known NaN pattern in tech_fitness.

    Countries A, B, C over years 2000-2004.
    - A: has leading NaN then values [NaN, NaN, 1.0, 2.0, 3.0]
    - B: all NaN (no patent data at all)
    - C: full data [5.0, 6.0, 7.0, 8.0, NaN] (trailing NaN)
    """
    years = [2000, 2001, 2002, 2003, 2004]
    countries = ['A', 'B', 'C']

    fitness = pd.DataFrame(
        [[1.0, 2.0, 3.0],
         [1.1, 2.1, 3.1],
         [1.2, 2.2, 3.2],
         [1.3, 2.3, 3.3],
         [1.4, 2.4, 3.4]],
        index=years, columns=countries,
    )
    gdp = pd.DataFrame(
        [[10.0, 20.0, 30.0],
         [10.1, 20.1, 30.1],
         [10.2, 20.2, 30.2],
         [10.3, 20.3, 30.3],
         [10.4, 20.4, 30.4]],
        index=years, columns=countries,
    )
    tech_fitness = pd.DataFrame(
        [[np.nan, np.nan, 5.0],
         [np.nan, np.nan, 6.0],
         [1.0,    np.nan, 7.0],
         [2.0,    np.nan, 8.0],
         [3.0,    np.nan, np.nan]],
        index=years, columns=countries,
    )

    return {'fitness': fitness, 'gdp': gdp, 'tech_fitness': tech_fitness}


class TestBaseline:
    def test_fills_with_global_min(self):
        obs = _make_observables()
        apply_imputation(obs, 'baseline')
        tf = obs['tech_fitness']
        global_min = 1.0  # min of all non-NaN values in original
        assert not tf.isna().any().any()
        assert tf.loc[2000, 'A'] == global_min
        assert tf.loc[2000, 'B'] == global_min
        assert tf.loc[2004, 'C'] == global_min

    def test_preserves_existing_values(self):
        obs = _make_observables()
        apply_imputation(obs, 'baseline')
        tf = obs['tech_fitness']
        assert tf.loc[2002, 'A'] == 1.0
        assert tf.loc[2000, 'C'] == 5.0


class TestDrop:
    def test_leaves_nan_in_tech_fitness(self):
        obs = _make_observables()
        apply_imputation(obs, 'drop')
        tf = obs['tech_fitness']
        # NaN cells should remain NaN (no imputation)
        assert tf.loc[2000, 'A'] != tf.loc[2000, 'A']  # NaN != NaN
        assert tf.loc[2000, 'B'] != tf.loc[2000, 'B']

    def test_preserves_existing_values(self):
        obs = _make_observables()
        apply_imputation(obs, 'drop')
        tf = obs['tech_fitness']
        assert tf.loc[2002, 'A'] == 1.0
        assert tf.loc[2003, 'A'] == 2.0


class TestCountryMin:
    def test_backfills_leading_nan(self):
        obs = _make_observables()
        apply_imputation(obs, 'country_min')
        tf = obs['tech_fitness']
        # A's earliest observed is 1.0 at 2002; leading NaN should be filled with 1.0
        assert tf.loc[2000, 'A'] == 1.0
        assert tf.loc[2001, 'A'] == 1.0

    def test_drops_all_nan_country(self):
        obs = _make_observables()
        apply_imputation(obs, 'country_min')
        tf = obs['tech_fitness']
        # B has no observations at all; should remain all NaN
        assert tf['B'].isna().all()

    def test_leaves_trailing_nan(self):
        obs = _make_observables()
        apply_imputation(obs, 'country_min')
        tf = obs['tech_fitness']
        # C's trailing NaN at 2004 should remain NaN (bfill only fills from later values)
        assert pd.isna(tf.loc[2004, 'C'])

    def test_preserves_existing_values(self):
        obs = _make_observables()
        apply_imputation(obs, 'country_min')
        tf = obs['tech_fitness']
        assert tf.loc[2002, 'A'] == 1.0
        assert tf.loc[2003, 'A'] == 2.0
        assert tf.loc[2000, 'C'] == 5.0
