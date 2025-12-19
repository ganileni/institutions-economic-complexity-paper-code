#!/usr/bin/env python
"""
Generate plots for the Polity paper.

This script generates all the figures used in the paper and SI appendix,
including error comparisons, model comparisons, country trajectories,
and the SPSb explanation figure.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
import numpy as np
import pycountry
from functools import partial
from collections import OrderedDict
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as pe

from ectools.NWKR import NWKR

from plotting import (
    make_fair_comparison,
    add_country_column,
    compute_prediction_errors,
    compute_signed_errors,
    take_only_last_n_periods,
    get_observables_dict_from_panel,
    plot_country_trajectories,
    format_plot,
    general_text_position_fixer,
)

# Countries to label in trajectory plots - with exact display names from original notebook
# This dict maps ISO3 country codes to their display names
INCLUDE_CTRS_DICT = {
    'AGO': 'Angola',
    'ARE': 'UA Emirates',
    'ARG': 'Argentina',
    'BHR': 'Bahrain',
    'BLR': 'Belarus',
    'BRA': 'Brazil',
    'CHE': 'Switzerland',
    'CHN': 'China',
    'COG': 'Congo',
    'CZE': 'Czechia',
    'DEU': 'Germany',
    'ETH': 'Ethiopia',
    'GBR': 'United Kingdom',
    'GEO': 'Georgia',
    'GHA': 'Ghana',
    'GIN': 'Guinea',
    'IDN': 'Indonesia',
    'IND': 'India',
    'IRN': 'Iran',
    'ISR': 'Israel',
    'ITA': 'Italy',
    'KOR': 'South Korea',
    'LBY': 'Libya',
    'LUX': 'Luxembourg',
    'MEX': 'Mexico',
    'MMR': 'Myanmar',
    'MNG': 'Mongolia',
    'MOZ': 'Mozambique',
    'NGA': 'Nigeria',
    'OMN': 'Oman',
    'PER': 'Peru',
    'POL': 'Poland',
    'QAT': 'Qatar',
    'RUS': 'Russia',
    'RWA': 'Rwanda',
    'SAU': 'Saudi Arabia',
    'SGP': 'Singapore',
    'SLE': 'Sierra Leone',
    'TGO': 'Togo',
    'TUN': 'Tunisia',
    'USA': 'United States',
    'UZB': 'Uzbekistan',
}
INCLUDE_CTRS_SET = set(INCLUDE_CTRS_DICT.keys())

# Model colors - exact same as original notebooks
MODELCOL = {
    'vspsb_fitness-gdp': '#ff7043',
    'vspsb_fitness-gdp-polity': '#0077bb',
    'vspsb_gdp-polity': '#009988',
    'vspsb_gdp': '#cc3311',
    'vspsb_polity': '#33bbee',
    'vspsb_fitness-gdp-polity-tech_fitness': '#ee3377',
    'vspsb_fitness-gdp-tech_fitness': '#8A00C5',
    'vspsb_gdp-tech_fitness': '#A0B710',
    'vspsb_gdp-polity-tech_fitness': '#585129',
    'autocorrelation_baseline': '#bbbbbb',
    'autocorrelation_baseline_mae': '#bbbbbb',
}
# Add non-vspsb versions
for key in list(MODELCOL):
    if key.startswith('vspsb'):
        MODELCOL[key.replace('vspsb_', '')] = MODELCOL[key]


def get_fixed_name(countrycode):
    """Get display-friendly country name from country code."""
    if countrycode in INCLUDE_CTRS_DICT:
        return INCLUDE_CTRS_DICT[countrycode]
    # Fallback to pycountry for countries not in our list
    try:
        return pycountry.countries.lookup(countrycode).name
    except LookupError:
        return countrycode


def rename_mae_vspsb(df):
    """Rename columns for consistency."""
    df.columns = [x.replace('vspsb_mae', 'mae_vspsb') for x in df.columns]
    df.columns = [x.replace('vspsb_ae', 'ae_vspsb') for x in df.columns]
    return df


def remove_vspsb_formatter(string):
    """Format model names for display."""
    return string.replace("vspsb_", "")


def add_custom_formatted_labels(custom_formatter, ax):
    """Add formatted labels to legend."""
    handles, labels = ax.get_legend_handles_labels()
    formatted_labels = [custom_formatter(label) for label in labels]
    return {"handles": handles, "labels": formatted_labels}


legendargs = partial(add_custom_formatted_labels, remove_vspsb_formatter)


def take_only_last_n_periods_extended(df, n_periods=5):
    """Extended version that handles varying dt values."""
    max_nyears = df['pred_dt'].max()
    last_year = df['year_pred_end'].max()
    last_n_years = set(list(last_year - x for x in range(max_nyears + n_periods)))
    trimmed = df.loc[df['year_pred_end'].isin(last_n_years), :]
    return trimmed




def savefig(fig, name, output_dir):
    """Save figure to output directory."""
    out_file = output_dir / f'{name}.pdf'
    out_file.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {name}.pdf")


def compute_pct_growth(year_start, country, df, observables):
    """Compute percentage growth from year_start."""
    gdp_vals = observables['gdp'][country]
    start_val = gdp_vals.loc[year_start]
    pct_growth = 10 ** (gdp_vals - start_val)
    # Only return data from year_start onwards
    pct_growth = pct_growth.loc[year_start:]
    return pct_growth


def get_forecast(year_start, country, model, df):
    """Get model forecast for a country starting from year_start."""
    df_sel = df.loc[(df.year_pred_start == year_start) & (df.country_code == country)]
    model_column = f'prediction_{model}' if 'autocorrelation' not in model else model
    df_sel = df_sel.set_index('year_pred_end')[model_column]
    df_sel.loc[year_start] = 0
    df_sel = df_sel.sort_index()
    increase_rates = 1 + df_sel
    years = df_sel.index - year_start
    increase_rates = increase_rates ** years.values
    return increase_rates


# =============================================================================
# MAIN FIGURE PLOTTING FUNCTIONS
# =============================================================================

def plot_error_comparison_global_avg(data, models, output_dir, nperiods=5):
    """Generate error comparison plot - global average over dt (SI Figure)."""
    include_models = [
        'vspsb_fitness-gdp-polity',
        'vspsb_gdp-polity',
        'vspsb_fitness-gdp',
        'vspsb_fitness-gdp-polity-tech_fitness',
        'vspsb_fitness-gdp-tech_fitness',
        'vspsb_gdp-tech_fitness',
        'vspsb_gdp-polity-tech_fitness',
        'vspsb_gdp',
    ]
    
    _scale = 0.66
    fig, ax = plt.subplots(figsize=(_scale * 10, _scale * 10))
    
    df = take_only_last_n_periods_extended(data, nperiods)
    means = 100 * df.select_dtypes(include=np.number).groupby('pred_dt').mean()
    
    for model in include_models:
        ax.plot(means.index, means[f'prediction_mae_{model}'], 
                color=MODELCOL[model], label=model, marker='.')
    ax.plot(means.index, means['autocorrelation_baseline_mae'], 
            label='autocorrelation_baseline', lw=3, color='black', linestyle='--')
    ax.set_ylabel('CAGR % MAE error')
    ax.set_xlabel(r'prediction $\Delta t$ (years)')
    ax.legend(**legendargs(ax))
    format_plot(ax)
    
    savefig(fig, f'si/error_comparison_all-global_avg-last_nperiods{nperiods}', output_dir)


def plot_error_distro_all(data, models, output_dir, nperiods=5, DT=4):
    """Generate error distribution plot for all models (SI Figure)."""
    df = take_only_last_n_periods_extended(data, nperiods)
    df = df.loc[df.pred_dt == DT]
    
    include_models = [
        'vspsb_fitness-gdp-polity',
        'vspsb_fitness-gdp-polity-tech_fitness',
        'vspsb_fitness-gdp-tech_fitness',
        'vspsb_gdp-tech_fitness',
        'vspsb_gdp-polity',
        'vspsb_fitness-gdp',
        'vspsb_gdp-polity-tech_fitness',
        'vspsb_gdp',
    ]
    include_models += ['autocorrelation_baseline_mae']
    
    fig, axes = plt.subplots(figsize=(7, 1.1 * len(include_models)), 
                             nrows=len(include_models), sharex=True, sharey=True)
    for ax, model in zip(axes, include_models):
        model_column = f"prediction_mae_{model}" if 'autocorrelation' not in model else model
        values = 100 * df[model_column].values
        values = values[np.isfinite(values)]
        ax.hist(values, bins=100, color=MODELCOL[model], label=model, 
                log=True, alpha=.5, density=True)
        median = np.nanmedian(values)
        ax.axvline(median, color='black', linestyle='dashed', 
                   label=f'median={round(median, 2)}')
        mean = np.nanmean(values)
        ax.axvline(mean, color='black', linestyle='solid', 
                   label=f'mean={round(mean, 2)}')
        ax.legend(**legendargs(ax))
        ax.set_ylabel('density')
        ax.set_xscale('log')
        ax.set_xticks([.01, .1, .5, 1, 5, 10, 15, 20])
        ax.set_xticklabels(['0.01', '0.1', '0.5', '1', '5', '10', '15', '20'])
        ax.set_xlim(.01, 20)
    axes[-1].set_xlabel('MAE CAGR % (log scale)')
    fig.suptitle("Distribution of errors for various SPSb models")
    fig.tight_layout()
    
    savefig(fig, f'si/error_distro-dt{DT}-all-last_nperiods{nperiods}', output_dir)


def plot_error_distro_select(data, models, output_dir, nperiods=5, DT=4):
    """Generate error distribution plot for selected models (Main Figure)."""
    df = take_only_last_n_periods_extended(data, nperiods)
    df = df.loc[df.pred_dt == DT]
    
    include_models = [
        'vspsb_fitness-gdp-polity-tech_fitness',
        'vspsb_gdp-tech_fitness',
        'vspsb_gdp-polity',
        'vspsb_fitness-gdp',
    ]
    
    fig, axes = plt.subplots(figsize=(7, 1.5 * len(include_models)), 
                             nrows=len(include_models), sharex=True, sharey=True)
    for ax, model in zip(axes, include_models):
        model_column = f"prediction_mae_{model}" if 'autocorrelation' not in model else model
        values = 100 * df[model_column].values
        values = values[np.isfinite(values)]
        ax.hist(values, bins=100, color=MODELCOL[model], label=model, 
                log=True, alpha=.5, density=True)
        median = np.nanmedian(values)
        ax.axvline(median, color='black', linestyle='dashed', 
                   label=f'median={round(median, 2)}')
        mean = np.nanmean(values)
        ax.axvline(mean, color='black', linestyle='solid', 
                   label=f'mean={round(mean, 2)}')
        ax.legend(**legendargs(ax))
        ax.set_ylabel('density')
        ax.set_xscale('log')
        ax.set_xticks([.01, .1, .5, 1, 5, 10, 15, 20])
        ax.set_xticklabels(['0.01', '0.1', '0.5', '1', '5', '10', '15', '20'])
        ax.set_xlim(.01, 20)
    axes[-1].set_xlabel('MAE CAGR % (log scale)')
    fig.tight_layout()
    
    savefig(fig, f'error_distro-dt{DT}-all-last_nperiods{nperiods}-select', output_dir)


def plot_trajectories_gdp_fitness(observables, output_dir, global_tflim, global_gdplim):
    """Generate trajectories in Fitness-GDP space (SI Figure)."""
    _scale = 0.66
    fig, ax = plt.subplots(figsize=(15 * _scale, 10 * _scale))
    
    plot_country_trajectories(
        ax=ax,
        xaxis=observables['fitness'].values,
        yaxis=observables['gdp'].values,
        names=observables['fitness'].columns,
        xlim=global_tflim, ylim=global_gdplim, title='',
        xlabel='', ylabel='')
    ax.set_ylabel('$log_{10}$(GDP)')
    ax.set_xlabel('$log_{10}$(Fitness)')
    ax.set_title('Trajectories')
    format_plot(ax)
    
    savefig(fig, 'si/trajectories_gdp_fitness', output_dir)


def plot_trajectories_gdp_polity2(observables, output_dir, global_politylim, global_gdplim):
    """Generate trajectories in Polity-GDP space (SI Figure)."""
    _scale = 0.66
    fig, ax = plt.subplots(figsize=(15 * _scale, 10 * _scale))
    
    plot_country_trajectories(
        ax=ax,
        xaxis=observables['polity'].values,
        yaxis=observables['gdp'].values,
        names=observables['fitness'].columns,
        xlim=global_politylim, ylim=global_gdplim, title='',
        xlabel='', ylabel='')
    ax.set_ylabel('$log_{10}$(GDP)')
    ax.set_xlabel('Polity 2')
    ax.set_title('Trajectories')
    format_plot(ax)
    
    savefig(fig, 'si/trajectories_gdp_polity2', output_dir)


def plot_trajectories_fitness_polity2(observables, output_dir):
    """Generate trajectories in Fitness-Polity space (SI Figure)."""
    _scale = 0.66
    fig, ax = plt.subplots(figsize=(15 * _scale, 10 * _scale))
    
    plot_country_trajectories(
        ax=ax,
        xaxis=observables['fitness'].values,
        yaxis=observables['polity'].values,
        names=observables['fitness'].columns,
        xlim=None, ylim=None, title='',
        xlabel='', ylabel='')
    ax.set_xlabel('$log_{10}$(Fitness)')
    ax.set_ylabel('Polity')
    ax.set_title('Trajectories')
    format_plot(ax)
    
    savefig(fig, 'si/trajectories_fitness_polity2', output_dir)


def plot_missing_analogues_when_techfit(observables, output_dir, global_tflim, global_gdplim, 
                                        tech_fitness_raw):
    """
    Generate plot showing where tech_fitness data is missing in Fitness-GDP space.
    
    Gray trajectories show country paths in Fitness-GDP space.
    Orange dots mark points where tech_fitness is NaN/missing.
    """
    _scale = 0.66
    fig, ax = plt.subplots(figsize=(15 * _scale, 10 * _scale))
    
    # Plot all trajectories in gray
    for country in observables['fitness'].columns:
        x = observables['fitness'][country].values
        y = observables['gdp'][country].values
        ax.plot(x, y, color='gray', alpha=0.5, linewidth=0.5)
    
    # Find missing tech_fitness points and plot them in orange
    for country in observables['fitness'].columns:
        if country not in tech_fitness_raw.columns:
            continue
        
        for year in observables['fitness'].index:
            if year not in tech_fitness_raw.index:
                continue
            
            tf_val = tech_fitness_raw.loc[year, country] if country in tech_fitness_raw.columns else np.nan
            
            if pd.isna(tf_val):
                # Get fitness and GDP for this country-year
                if year in observables['fitness'].index and country in observables['fitness'].columns:
                    fit_val = observables['fitness'].loc[year, country]
                    gdp_val = observables['gdp'].loc[year, country]
                    
                    if np.isfinite(fit_val) and np.isfinite(gdp_val):
                        ax.scatter(fit_val, gdp_val, color='orange', s=5, alpha=0.7, zorder=5)
    
    ax.set_xlabel('$log_{10}$(Fitness)')
    ax.set_ylabel('$log_{10}$(GDP)')
    ax.set_title('Missing tech_fitness values (orange dots)')
    ax.set_xlim(global_tflim)
    ax.set_ylim(global_gdplim)
    format_plot(ax)
    
    savefig(fig, 'si/missing_analogues_when_techfit', output_dir)


def plot_country_relative_error_comparison(data, models, country, output_dir, DT=4):
    """Generate relative error comparison for a country (Main Figure)."""
    df = data.loc[(data.country_code == country) & (data.pred_dt == DT)].sort_values(by='year_pred_start')
    
    if country == 'CHN':
        included_models = ['fitness-gdp-polity', 'fitness-gdp', 
                          'fitness-gdp-polity-tech_fitness', 'gdp']
    else:  # IND
        included_models = ['gdp-polity', 'fitness-gdp', 
                          'fitness-gdp-polity-tech_fitness', 'gdp-tech_fitness',
                          'gdp-polity-tech_fitness']
    
    _scale = 1.2
    fig, ax = plt.subplots(figsize=(_scale * 4, _scale * 3))
    ax.axhline(0, color='black', linestyle='solid', linewidth=0.5)
    
    for model in models:
        if model not in included_models:
            continue
        ax.plot(df.year_pred_start, 100 * df[f'prediction_ae_vspsb_{model}'], 
                label='vspsb_' + model, marker='.', color=MODELCOL[model])
    
    if country == 'CHN':
        ax.plot(df.year_pred_start, 100 * df['autocorrelation_baseline_ae'],
                label='autocorrelation_baseline', marker='.', color='black', linestyle='dashed')
    
    ax.legend(**legendargs(ax), framealpha=.5)
    ax.set_title(rf'Errors for a $\Delta t$={DT} prediction of {country} starting in...')
    ax.set_xlabel('Year')
    ax.set_ylabel('[Model prediction] - [Ground Truth]')
    format_plot(ax)
    fig.tight_layout()
    
    savefig(fig, f'country_{country}_relative_error_comparison', output_dir)


def plot_country_forecasts(data, models, country, year_start, observables, output_dir):
    """Generate forecast comparison for a country (Main Figure)."""
    df = data
    
    if country == 'CHN':
        included_models = ['fitness-gdp-polity', 'fitness-gdp', 
                          'fitness-gdp-polity-tech_fitness', 'gdp']
    else:  # IND
        included_models = ['gdp-polity', 'fitness-gdp', 
                          'fitness-gdp-polity-tech_fitness', 'gdp-tech_fitness',
                          'gdp-polity-tech_fitness']
    
    _scale = 1.2
    pct_growth = compute_pct_growth(year_start, country, df, observables)
    fig, ax = plt.subplots(figsize=(_scale * 4, _scale * 3))
    
    for model in included_models:
        forecast = get_forecast(year_start, country, model, df)
        ax.plot(forecast.index, 100 * forecast, label=model, marker='.', 
                color=MODELCOL[model])
    
    ax.plot(pct_growth.index, 100 * pct_growth, color='black', linestyle='solid', 
            linewidth=1, label='ground truth')
    ax.legend(**legendargs(ax))
    ax.set_ylabel(f"GDP (100 = {year_start}'s value)")
    ax.set_xlabel('Year')
    ax.set_title(f'Models prediction for {country}')
    
    # Set x-axis limits to start at year_start
    ax.set_xlim(year_start, pct_growth.index.max())
    
    # Format x-axis with integer years, tilted 45 degrees
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    format_plot(ax)
    fig.tight_layout()
    
    savefig(fig, f'country_{country}_forecasts_start_{year_start}', output_dir)


def plot_whichbest_generic(data, observables, output_dir, nperiods, DT, 
                           included_models, colormap, figure_name,
                           global_xlim, global_ylim, legend_title='Lowest error\nachieved by',
                           legend_entries=None, plane_type='fitness_gdp', polity_jitter=None):
    """
    Generic function to plot which model is best in a given plane.
    
    Parameters:
    -----------
    legend_entries : list of tuples, optional
        If provided, use these entries for the legend instead of model names.
        Each tuple should be (name, color) where name is the display text.
    plane_type : str
        'fitness_gdp' for fitness-GDP plane (default)
        'polity_gdp' for polity-GDP plane (uses dots with jitter)
    polity_jitter : pd.DataFrame, optional
        Jitter values to add to polity coordinates (required for polity_gdp plane)
    """
    df = data
    df = take_only_last_n_periods_extended(df, nperiods)
    df = df.loc[df.pred_dt == DT, :]
    
    _scale = 0.66
    fig, ax = plt.subplots(figsize=(15 * _scale, 10 * _scale))
    
    patches_alpha = 0.3
    resolution = 30
    
    xpixelsize = (global_xlim[1] - global_xlim[0]) / resolution
    ypixelsize = (global_ylim[1] - global_ylim[0]) / resolution
    
    bwscale = 3
    bw = xpixelsize * bwscale, ypixelsize * bwscale
    
    # Select X values based on plane type
    if plane_type == 'polity_gdp':
        X = df[['value_polity', 'value_gdp']].values
        x_observable = 'polity'
        xlabel = 'Polity'
    else:
        X = df[['value_fitness', 'value_gdp']].values
        x_observable = 'fitness'
        xlabel = r'$log_{10}$(Fitness)'
    
    x0basepositions = np.linspace(*global_xlim, resolution)
    x1basepositions = np.linspace(*global_ylim, resolution)
    coords = np.meshgrid(x0basepositions, x1basepositions)
    nwkr_coords = np.stack([coords[0].flatten(), coords[1].flatten()]).T
    
    results_pred = {}
    for modelname in included_models:
        y = df[f'prediction_mae_vspsb_{modelname}'].values
        model = NWKR(bandwidth=bw)
        model.fit(X=X, y=y)
        _pred, _std = model.predict(nwkr_coords)
        results_pred[modelname] = _pred
    
    stacked = np.vstack([results_pred[m] for m in included_models])
    whichmin = np.argmin(stacked, axis=0)
    colors = [colormap[m] for m in included_models]
    
    for x, y, c in zip(coords[0].flatten(), coords[1].flatten(), whichmin):
        pixel = mpatches.Rectangle(
            xy=(x, y),
            width=xpixelsize,
            height=ypixelsize,
            fill=True,
            edgecolor='none',
            alpha=patches_alpha,
            linewidth=0,
            facecolor=colors[c]
        )
        ax.add_patch(pixel)
    
    # Plot labeled trajectories/dots
    names = [name for name in observables['fitness'].columns if name in INCLUDE_CTRS_SET]
    
    if plane_type == 'polity_gdp' and polity_jitter is not None:
        # Polity plane: use scatter points with jitter (no trajectory lines)
        xaxis_labeled = observables['polity'][names].values + polity_jitter[names].values
        xaxis_all = observables['polity'].values + polity_jitter.values
        use_scatter_only = True
    else:
        # Fitness plane: use trajectories
        xaxis_labeled = observables[x_observable][names].values
        xaxis_all = observables[x_observable].values
        use_scatter_only = False
    
    annotations = plot_country_trajectories(
        ax=ax,
        xaxis=xaxis_labeled,
        yaxis=observables['gdp'][names].values,
        names=[get_fixed_name(name) for name in observables['fitness'].columns if name in INCLUDE_CTRS_SET],
        scatter_only=use_scatter_only,
    )
    
    # Plot all trajectories/dots
    plot_country_trajectories(
        ax=ax,
        xaxis=xaxis_all,
        yaxis=observables['gdp'].values,
        names=["" for name in observables['fitness'].columns],
        traj_colors=False,
        traj_alpha=.3,
        scatter_only=use_scatter_only,
    )
    
    ax.set_ylabel(r'$log_{10}$(GDP)')
    ax.set_xlabel(xlabel)
    ax.set_xlim(global_xlim[0], global_xlim[1])
    ax.set_ylim(global_ylim[0], global_ylim[1])
    
    # Apply text position fixer if there are annotations
    if annotations:
        general_text_position_fixer(fig=fig, ax=ax, texts=annotations)
    
    # Create legend
    divider = make_axes_locatable(ax)
    legend_ax = divider.append_axes("right", size="15%", pad=0.1)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(False)
    
    # Use custom legend entries if provided, otherwise use colormap keys
    if legend_entries is not None:
        legend_items = legend_entries
    else:
        legend_items = [(key, item) for key, item in colormap.items() if key in included_models]
    
    num_items = len(legend_items)
    delta = 0.1
    spacing = (1 - delta) / (num_items + 1)
    
    legend_ax.set_title(legend_title, fontsize='medium', y=0.90, x=1.1)
    
    for i, (name, color) in enumerate(legend_items):
        y_position = 1 - delta - spacing * (i + 1)
        patch = mpatches.Rectangle((0.1, y_position), 0.2, spacing * 0.8, 
                                   facecolor=color, edgecolor=None, alpha=patches_alpha)
        legend_ax.add_patch(patch)
        legend_ax.text(0.35, y_position + spacing * 0.4, name, va='center', ha='left')
    
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    ax.set_title(f'Best predictor model for each cell of the {ax.get_xlabel()}-{ax.get_ylabel()} plane')
    
    savefig(fig, figure_name, output_dir)


def main():
    # Paths relative to this script location
    script_dir = Path(__file__).parent
    data_dir = script_dir / '../data'
    predictions_dir = script_dir / '../output/predictions'
    output_dir = script_dir / '../output/plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'si').mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    predictions_file = predictions_dir / 'polity-short-4d-backfill.csv'
    if not predictions_file.exists():
        print(f"Error: Predictions file not found at {predictions_file}")
        print("Please run run_predictions.py first.")
        return
    
    print(f"Loading predictions from: {predictions_file}")
    data = pd.read_csv(predictions_file)
    
    # Get list of models
    models = [
        x.replace('prediction_', '') 
        for x in data.columns 
        if 'prediction' in x and 'vspsb' not in x
    ]
    print(f"Found {len(models)} models: {models}")
    
    # Process data
    data = make_fair_comparison(data, models)
    data = add_country_column(data)
    compute_prediction_errors(data, models)
    compute_signed_errors(data, models)
    data = rename_mae_vspsb(data)
    
    print(f"Data shape after processing: {data.shape}")
    
    # Get observables for trajectory plots
    observables = get_observables_dict_from_panel(data)
    
    # Load raw tech_fitness for missing analogues plot
    tech_fitness_raw = pd.read_csv(data_dir / 'PATSTAT_tech_fitness/tech_fitness_8dig.csv', index_col=0)
    
    # Define global limits
    nperiods = 5
    DT = 4
    df_for_limits = take_only_last_n_periods_extended(data, nperiods)
    df_for_limits = df_for_limits.loc[df_for_limits.pred_dt == DT, :]
    
    global_tflim = (-4, 1.5)
    global_gdplim = (2.7, 5)
    global_politylim = (df_for_limits['value_polity'].min(), 
                        df_for_limits['value_polity'].max() + 0.25)
    
    # Generate polity jitter for polity-plane figures
    np.random.seed(42)
    polity_jitter = np.random.rand(*observables['polity'].values.shape) * 0.14
    polity_jitter[0, :] = 0.  # Don't jitter start of trajectory
    polity_jitter_df = observables['polity'].copy()
    polity_jitter_df.loc[:, :] = polity_jitter
    
    print("\nGenerating plots...")
    
    # ==========================================================================
    # SI FIGURES
    # ==========================================================================
    
    # Error comparison global average
    plot_error_comparison_global_avg(data, models, output_dir, nperiods=5)
    
    # Error distribution - all models
    plot_error_distro_all(data, models, output_dir, nperiods=5, DT=4)
    
    # Trajectory plots
    plot_trajectories_gdp_fitness(observables, output_dir, global_tflim, global_gdplim)
    plot_trajectories_gdp_polity2(observables, output_dir, global_politylim, global_gdplim)
    plot_trajectories_fitness_polity2(observables, output_dir)
    
    # Missing analogues when techfit
    plot_missing_analogues_when_techfit(observables, output_dir, global_tflim, global_gdplim,
                                        tech_fitness_raw)
    
    # Which best - all models (SI)
    included_all = ['fitness-gdp-polity', 'gdp-polity', 'fitness-gdp',
                    'fitness-gdp-polity-tech_fitness', 'fitness-gdp-tech_fitness',
                    'gdp-tech_fitness', 'gdp-polity-tech_fitness']
    plot_whichbest_generic(data, observables, output_dir, nperiods, DT,
                          included_all, MODELCOL, 'si/dt4_whichbest_tf_all',
                          global_tflim, global_gdplim)
    
    # Which best - 2d3d comparison (SI)
    included_2d3d = ['fitness-gdp-polity', 'gdp-polity']
    plot_whichbest_generic(data, observables, output_dir, nperiods, DT,
                          included_2d3d, MODELCOL, 'si/dt4_whichbest_2d3d',
                          global_tflim, global_gdplim)
    
    # Which best - 3d3d comparison (SI) - fitness-gdp-polity vs fitness-gdp-tech_fitness
    included_3d3d = ['fitness-gdp-polity', 'fitness-gdp-tech_fitness']
    plot_whichbest_generic(data, observables, output_dir, nperiods, DT,
                          included_3d3d, MODELCOL, 'si/dt4_whichbest_tf_3d3d',
                          global_tflim, global_gdplim)
    
    # Which best - 3d3d 2var change (SI) - fitness-gdp-tech_fitness vs gdp-polity-tech_fitness
    included_3d3d_2var = ['fitness-gdp-tech_fitness', 'gdp-polity-tech_fitness']
    plot_whichbest_generic(data, observables, output_dir, nperiods, DT,
                          included_3d3d_2var, MODELCOL, 'si/dt4_whichbest_tf_3d3d_2varchange',
                          global_tflim, global_gdplim)
    
    # ==========================================================================
    # MAIN FIGURES
    # ==========================================================================
    
    # Error distribution - select models
    plot_error_distro_select(data, models, output_dir, nperiods=5, DT=4)
    
    # Country error comparisons
    plot_country_relative_error_comparison(data, models, 'CHN', output_dir, DT=4)
    plot_country_relative_error_comparison(data, models, 'IND', output_dir, DT=4)
    
    # Country forecasts
    plot_country_forecasts(data, models, 'CHN', 2006, observables, output_dir)
    plot_country_forecasts(data, models, 'IND', 2006, observables, output_dir)
    
    # Which best - 2d2d comparison (Main)
    included_2d2d = ['gdp-polity', 'fitness-gdp']
    plot_whichbest_generic(data, observables, output_dir, nperiods, DT,
                          included_2d2d, MODELCOL, 'dt4_whichbest_2d2d',
                          global_tflim, global_gdplim)
    
    # Which best - tf 2d2d2d (Main)
    included_tf_2d2d2d = ['gdp-polity', 'fitness-gdp', 'gdp-tech_fitness']
    plot_whichbest_generic(data, observables, output_dir, nperiods, DT,
                          included_tf_2d2d2d, MODELCOL, 'dt4_whichbest_tf_2d2d2d',
                          global_tflim, global_gdplim)
    
    # Aggregate coloring - base + polity vs tech_fitness
    _polity = '#ff7043'
    _techfit = '#0077bb'
    _both = '#009988'
    _none = '#BB00C1'
    
    aggregate_colors = {
        'fitness-gdp': _none,
        'fitness-gdp-polity': _polity,
        'gdp-polity': _polity,
        'fitness-gdp-polity-tech_fitness': _both,
        'fitness-gdp-tech_fitness': _techfit,
        'gdp-tech_fitness': _techfit,
        'gdp-polity-tech_fitness': _both,
    }
    # Aggregate legend entries
    aggregate_legend = [
        ('base\n(gdp-fitness only)', _none),
        ('base\n+ polity', _polity),
        ('base\n+ tech_fitness', _techfit),
        ('base\n+ polity\n+ tech_fitness', _both),
    ]
    plot_whichbest_generic(data, observables, output_dir, nperiods, DT,
                          included_all, aggregate_colors, 'dt4_whichbest_tf_all_aggregate',
                          global_tflim, global_gdplim, 
                          legend_title='Lowest error\nmodel',
                          legend_entries=aggregate_legend)
    
    # By dimensionality coloring
    _4D = '#ff7043'
    _3D = '#0077bb'
    _2D = '#009988'
    
    dimension_colors = {
        'fitness-gdp': _2D,
        'fitness-gdp-polity': _3D,
        'gdp-polity': _2D,
        'fitness-gdp-polity-tech_fitness': _4D,
        'fitness-gdp-tech_fitness': _3D,
        'gdp-tech_fitness': _2D,
        'gdp-polity-tech_fitness': _3D,
    }
    # Dimension legend entries
    dimension_legend = [
        ('4D', _4D),
        ('3D', _3D),
        ('2D', _2D),
    ]
    plot_whichbest_generic(data, observables, output_dir, nperiods, DT,
                          included_all, dimension_colors, 'dt4_whichbest_tf_all_bydimension',
                          global_tflim, global_gdplim,
                          legend_title='Lowest error\nmodel',
                          legend_entries=dimension_legend)
    
    # Polity 2d2d comparison (Main) - in polity-GDP plane with dots and jitter
    included_polity_2d2d = ['gdp-polity', 'fitness-gdp']
    plot_whichbest_generic(data, observables, output_dir, nperiods, DT,
                          included_polity_2d2d, MODELCOL, 'dt4_whichbest_polity_2d2d',
                          global_politylim, global_gdplim,
                          plane_type='polity_gdp', polity_jitter=polity_jitter_df)
    
    # ==========================================================================
    # SI POLITY FIGURES - All in polity-GDP plane with dots and jitter
    # ==========================================================================
    
    # Polity tf all (SI)
    included_polity_all = ['fitness-gdp-polity', 'gdp-polity', 'fitness-gdp',
                           'fitness-gdp-polity-tech_fitness', 'fitness-gdp-tech_fitness',
                           'gdp-tech_fitness', 'gdp-polity-tech_fitness']
    plot_whichbest_generic(data, observables, output_dir, nperiods, DT,
                          included_polity_all, MODELCOL, 'si/dt4_whichbest_tf_polity_all',
                          global_politylim, global_gdplim,
                          plane_type='polity_gdp', polity_jitter=polity_jitter_df)
    
    # Polity 2d2d2d (SI)
    included_polity_2d2d2d = ['gdp-polity', 'fitness-gdp', 'gdp-tech_fitness']
    plot_whichbest_generic(data, observables, output_dir, nperiods, DT,
                          included_polity_2d2d2d, MODELCOL, 'si/dt4_whichbest_tf_polity_2d2d2d',
                          global_politylim, global_gdplim,
                          plane_type='polity_gdp', polity_jitter=polity_jitter_df)
    
    # Polity aggregate (SI)
    plot_whichbest_generic(data, observables, output_dir, nperiods, DT,
                          included_polity_all, aggregate_colors, 'si/dt4_whichbest_tf_polity_all_aggregate',
                          global_politylim, global_gdplim,
                          legend_title='Lowest error\nmodel',
                          legend_entries=aggregate_legend,
                          plane_type='polity_gdp', polity_jitter=polity_jitter_df)
    
    # Polity by dimension (SI)
    plot_whichbest_generic(data, observables, output_dir, nperiods, DT,
                          included_polity_all, dimension_colors, 'si/dt4_whichbest_tf_polity_all_bydimension',
                          global_politylim, global_gdplim,
                          legend_title='Lowest error\nmodel',
                          legend_entries=dimension_legend,
                          plane_type='polity_gdp', polity_jitter=polity_jitter_df)
    
    print(f"\nAll plots saved to: {output_dir}")
    print("\nNote: SPSb explanation figure and bootstrap statistical figures")
    print("require additional code and longer computation time.")
    print("Please run generate_stats.py for statistical analysis figures.")


if __name__ == '__main__':
    main()
