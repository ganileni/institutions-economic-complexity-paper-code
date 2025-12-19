#!/usr/bin/env python
"""
Generate statistical analysis plots for the Polity paper.

This script generates:
1. Bootstrap analysis figures comparing model dimensionalities
2. SPSb explanation figure showing how the method works

WARNING: Bootstrap analysis can take a long time (~30 min per figure with n_bootstrap=500).
         Set n_bootstrap to a smaller value (e.g., 50) for testing.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import pycountry
from functools import partial
from collections import OrderedDict
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

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
)

# Countries to label in trajectory plots - with exact display names from original notebook
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

# Model colors
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
    df.columns = [x.replace('vspsb_mae', 'mae_vspsb') for x in df.columns]
    df.columns = [x.replace('vspsb_ae', 'ae_vspsb') for x in df.columns]
    return df


def take_only_last_n_periods_extended(df, n_periods=5):
    max_nyears = df['pred_dt'].max()
    last_year = df['year_pred_end'].max()
    last_n_years = set(list(last_year - x for x in range(max_nyears + n_periods)))
    return df.loc[df['year_pred_end'].isin(last_n_years), :]


def savefig(fig, name, output_dir):
    out_file = output_dir / f'{name}.pdf'
    out_file.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_file, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {name}.pdf")


def get_delta(o, dt, deltatype='diff'):
    """Compute delta (change) over dt years."""
    if deltatype == 'diff':
        deltas = o.iloc[dt:, :].values - o.iloc[:-dt, :].values
    elif deltatype == 'cagr':
        deltas = 100 * (((o.iloc[dt:, :].values / o.iloc[:-dt, :].values) ** (1/dt)) - 1)
    else:
        raise ValueError(f"Unknown deltatype: {deltatype}")
    deltas_df = o.copy().iloc[:-dt, :]
    deltas_df.iloc[:, :] = deltas
    return deltas_df


def turn_off_ax_labels_ticks(ax):
    """Turn off axis labels and ticks."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.tick_params(labelcolor='none')


def plot_spsb_explanation(data, output_dir, seed=41):
    """
    Generate the SPSb explanation figure showing how the method works.
    
    This creates a 3-panel figure:
    - Left: All analogues and their velocity arrows
    - Middle: Bootstrap sampling illustration
    - Right: Final prediction from averaged bootstrap samples
    """
    country = 'BRA'
    year = 2008
    dt = 1
    
    observables = get_observables_dict_from_panel(data)
    observables_delta = {key: get_delta(value, dt=dt) for key, value in observables.items()}
    observables = {key: value.iloc[:-dt, :] for key, value in observables.items()}
    
    xobs = observables['fitness']
    yobs = observables['gdp']
    xdeltas = observables_delta['fitness']
    ydeltas = observables_delta['gdp']
    yobsname = r'$\log_{10}(GDPpc)$'
    xobsname = r'$\log_{10}(Fitness)$'
    
    rangescale = 0.5
    xrange = 0.5 * rangescale
    yrange = 1 * rangescale
    
    cname = pycountry.countries.lookup(country).name
    
    past = xobs.index.values <= year
    future = ~past
    future[xobs.index.values == year] = True
    
    xpos = xobs.loc[year, country]
    ypos = yobs.loc[year, country]
    
    np.random.seed(seed)
    
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    
    ax_left = fig.add_axes([0.05, 0.1, 0.3, 0.8])
    
    mid_axes = []
    for i in range(4):
        ax = fig.add_axes([0.37, 0.70 - i * 0.2, 0.25, 0.19])
        turn_off_ax_labels_ticks(ax)
        mid_axes.append(ax)
    
    ax_right = fig.add_axes([0.66, 0.1, 0.3, 0.8])
    
    # ========== Left panel: All analogues ==========
    ax = ax_left
    
    ax.scatter([xpos], [ypos], marker='x', s=50, color='red', 
               label=f'{cname} in {year} (state to predict)')
    
    for col in xobs.columns:
        ax.plot(xobs[col], yobs[col], linewidth=0.5, color='black', alpha=0.3)
    
    ax.scatter(xobs.loc[future, :].values.flatten(), 
               yobs.loc[future, :].values.flatten(), 
               marker='.', s=1, color='red', alpha=0.5, 
               label='future analogues (no backtesting)')
    
    X = xobs.loc[past, :].values.flatten()
    Y = yobs.loc[past, :].values.flatten()
    U = xdeltas.loc[past, :].values.flatten()
    V = ydeltas.loc[past, :].values.flatten()
    
    keep = np.isfinite(X * Y * U * V)
    
    arrowprops = dict(
        headwidth=3,
        headlength=5,
        headaxislength=4.5,
        width=0.005,
        facecolor='green',
        alpha=0.5,
        zorder=1,
        scale=1,
    )
    arrowscale = 1
    
    ax.quiver(X[keep], Y[keep], U[keep] * arrowscale, V[keep] * arrowscale,
              angles='xy', scale_units='xy', **arrowprops)
    
    ax.plot(xobs.loc[past, country], yobs.loc[past, country], 
            color='red', label=f'{cname} past trajectory')
    ax.plot(xobs.loc[future, country], yobs.loc[future, country], 
            color='blue', label=f'{cname} future trajectory')
    
    ax.set_xlim(xpos - xrange, xpos + xrange)
    ax.set_ylim(ypos - yrange, ypos + yrange)
    ax.set_title(f"Analogues for {cname} in {year}")
    ax.set_xlabel(xobsname)
    ax.set_ylabel(yobsname)
    
    handles, labels = ax.get_legend_handles_labels()
    arrow_legend = mpatches.FancyArrowPatch(
        posA=(0, 0), posB=(0.5, 0.5), 
        arrowstyle='simple', facecolor='green', alpha=0.5
    )
    handles.append(arrow_legend)
    labels.append(f'{cname} past analogues')
    ax.legend(handles=handles, labels=labels)
    
    # ========== Middle panels: Bootstrap sampling ==========
    mid_axes[0].set_title('Bootstrap\n(not to scale)')
    
    n_samples = 10
    sampling_prob = ((X - xpos)**2 + (Y - ypos)**2)**0.5
    sampling_prob[~np.isfinite(sampling_prob)] = 0.
    sampling_prob /= sampling_prob.sum()
    samples = [np.random.choice(list(range(X.size)), replace=False, 
                                size=n_samples, p=sampling_prob) 
               for _ in mid_axes]
    
    avg_arrowprops = dict(**arrowprops)
    avg_arrowprops.update({
        'width': arrowprops['width'] * 2,
        'headwidth': arrowprops['headwidth'] * 2,
        'facecolor': 'purple',
        'scale': arrowprops['scale'] * 0.5
    })
    
    avgd_us = []
    avgd_vs = []
    
    for ax, samp in zip(mid_axes[:-1], samples):
        xsamplepos = xpos + np.random.normal(scale=0.5, size=len(samp)) * xrange
        ysamplepos = ypos + np.random.normal(scale=0.5, size=len(samp)) * yrange
        
        _country_pos = ax.scatter([xpos], [ypos], marker='x', s=50, color='red')
        _sampled_analogues = ax.quiver(
            xsamplepos, ysamplepos,
            U[samp] * arrowscale, V[samp] * arrowscale,
            angles='xy', scale_units='xy', **arrowprops
        )
        
        avgU = U[samp].mean()
        avgV = V[samp].mean()
        _average_of_samples = ax.quiver(
            [xpos], [ypos], [avgU], [avgV],
            angles='xy', scale_units='xy', **avg_arrowprops
        )
        
        avgd_us.append(avgU)
        avgd_vs.append(avgV)
        ax.set_xlim(xpos - xrange, xpos + xrange)
        ax.set_ylim(ypos - yrange, ypos + yrange)
    
    # Legend in last mid axis
    ax = mid_axes[-1]
    turn_off_ax_labels_ticks(ax)
    ax.legend([_country_pos, _sampled_analogues, _average_of_samples],
              [f'{cname} in {year}', 'Sampled analogues', 'Average of sampled analogues'],
              loc='center')
    ax.set_axis_off()
    
    # ========== Right panel: Final prediction ==========
    ax = ax_right
    
    ax.scatter([xpos], [ypos], marker='x', s=50, color='red',
               label=f'{cname} in {year} (state to predict)')
    
    for col in xobs.columns:
        ax.plot(xobs[col], yobs[col], linewidth=0.5, color='black', alpha=0.3)
    
    sampled_arrowprops = dict(**arrowprops)
    sampled_arrowprops.update(facecolor=avg_arrowprops['facecolor'])
    ax.quiver(
        [xpos] * len(avgd_us), [ypos] * len(avgd_us),
        avgd_us, avgd_vs,
        angles='xy', scale_units='xy', label='bootstrap averages',
        **sampled_arrowprops
    )
    
    final_pred_arrowprops = dict(**avg_arrowprops)
    final_pred_arrowprops.update(dict(facecolor='red'))
    ax.quiver(
        [xpos], [ypos],
        np.mean(avgd_us), np.mean(avgd_vs),
        angles='xy', scale_units='xy', label='SPSb prediction',
        **final_pred_arrowprops
    )
    
    ax.plot(xobs.loc[past, country], yobs.loc[past, country], 
            color='red', label=f'{cname} past trajectory', alpha=0.3)
    ax.plot(xobs.loc[future, country], yobs.loc[future, country], 
            color='blue', label=f'{cname} future trajectory', alpha=0.3)
    
    ax.set_xlim(xobs.loc[:, country].min(), xobs.loc[:, country].max())
    ax.set_ylim(yobs.loc[:, country].min(), yobs.loc[:, country].max())
    ax.set_title(f"Prediction for {cname} in {year}")
    ax.set_xlabel(xobsname)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='lower center')
    
    savefig(fig, f'spsb_explanation_{country}_{year}_seed{seed}', output_dir)


def bootstrap_single_replicate(b, X, model_y_data, bw, nwkr_coords, included_models):
    """Perform a single bootstrap replicate."""
    np.random.seed(b)
    n = len(X)
    bootstrap_indices = np.random.choice(n, size=n, replace=True)
    X_boot = X[bootstrap_indices]
    
    boot_predictions = {}
    for modelname in included_models:
        y_model_boot = model_y_data[modelname][bootstrap_indices]
        model = NWKR(bandwidth=bw)
        model.fit(X=X_boot, y=y_model_boot)
        _pred, _ = model.predict(nwkr_coords)
        boot_predictions[modelname] = _pred
    
    return boot_predictions


def plot_bootstrap_2models(data, observables, output_dir, nperiods, DT,
                           model0, model1, n_bootstrap=500, n_jobs=4):
    """
    Generate bootstrap comparison plot for 2 models.
    """
    print(f"\nGenerating bootstrap comparison: {model0} vs {model1}")
    print(f"Using {n_bootstrap} bootstrap replicates on {n_jobs} cores...")
    
    df = data.copy()
    df = take_only_last_n_periods_extended(df, nperiods)
    df = df.loc[df.pred_dt == DT, :]
    
    _scale = 0.66
    global_gdplim = (2.7, 5)
    global_tflim = (-4, 1.5)
    global_ylim = global_gdplim
    global_xlim = global_tflim
    patches_alpha = 0.3
    resolution = 30
    
    xpixelsize = (global_xlim[1] - global_xlim[0]) / resolution
    ypixelsize = (global_ylim[1] - global_ylim[0]) / resolution
    
    bwscale = 3.5
    bw = xpixelsize * bwscale, ypixelsize * bwscale
    
    X = df[['value_fitness', 'value_gdp']].values
    
    x0basepositions = np.linspace(*global_xlim, resolution)
    x1basepositions = np.linspace(*global_ylim, resolution)
    coords = np.meshgrid(x0basepositions, x1basepositions)
    nwkr_coords = np.stack([coords[0].flatten(), coords[1].flatten()]).T
    
    included_models = [model0, model1]
    
    # Fit original models
    results_pred = {}
    for modelname in included_models:
        y = df[f'prediction_mae_vspsb_{modelname}'].values
        model = NWKR(bandwidth=bw)
        model.fit(X=X, y=y)
        _pred, _ = model.predict(nwkr_coords)
        results_pred[modelname] = _pred
    
    stacked = np.vstack([results_pred[m] for m in included_models])
    whichmin = np.argmin(stacked, axis=0)
    
    # Bootstrap
    model_y_data = {m: df[f'prediction_mae_vspsb_{m}'].values for m in included_models}
    
    print("Running bootstrap...")
    bootstrap_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(bootstrap_single_replicate)(b, X, model_y_data, bw, nwkr_coords, included_models)
        for b in range(n_bootstrap)
    )
    
    print("Computing p-values...")
    n_pixels = len(nwkr_coords)
    bootstrap_pvalues = np.zeros(n_pixels)
    bootstrap_win_rate_model0 = np.zeros(n_pixels)
    
    for pixel_idx in range(n_pixels):
        model0_errors = np.array([bootstrap_results[b][model0][pixel_idx] for b in range(n_bootstrap)])
        model1_errors = np.array([bootstrap_results[b][model1][pixel_idx] for b in range(n_bootstrap)])
        
        valid = ~(np.isnan(model0_errors) | np.isnan(model1_errors) |
                  np.isinf(model0_errors) | np.isinf(model1_errors))
        
        if np.sum(valid) == 0:
            bootstrap_pvalues[pixel_idx] = np.nan
            bootstrap_win_rate_model0[pixel_idx] = np.nan
            continue
        
        model0_errors = model0_errors[valid]
        model1_errors = model1_errors[valid]
        
        model0_wins = np.sum(model0_errors < model1_errors)
        bootstrap_win_rate_model0[pixel_idx] = model0_wins / len(model0_errors)
        
        model1_wins = np.sum(model1_errors < model0_errors)
        p_model0_better = model0_wins / len(model0_errors)
        p_model1_better = model1_wins / len(model0_errors)
        
        bootstrap_pvalues[pixel_idx] = 2 * min(p_model0_better, p_model1_better)
        bootstrap_pvalues[pixel_idx] = min(bootstrap_pvalues[pixel_idx], 1.0)
    
    # FDR correction
    fdr_level = 0.05
    valid_mask = ~np.isnan(bootstrap_pvalues)
    valid_pvalues = bootstrap_pvalues[valid_mask]
    
    reject, _, _, _ = multipletests(valid_pvalues, alpha=fdr_level, method='fdr_bh')
    bh_threshold = np.max(valid_pvalues[reject]) if np.any(reject) else 0.05
    
    reject_full = np.zeros(n_pixels, dtype=bool)
    reject_full[valid_mask] = reject
    
    # Plotting
    fig_combined, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12 * _scale, 20 * _scale))
    
    colors = [MODELCOL[m] for m in included_models]
    
    # Top panel: which model wins
    for x, y, c in zip(coords[0].flatten(), coords[1].flatten(), whichmin):
        pixel = mpatches.Rectangle(
            xy=(x, y), width=xpixelsize, height=ypixelsize,
            fill=True, edgecolor='none', alpha=patches_alpha,
            linewidth=0, facecolor=colors[c]
        )
        ax_top.add_patch(pixel)
    
    names = [name for name in observables['fitness'].columns if name in INCLUDE_CTRS_SET]
    plot_country_trajectories(
        ax=ax_top,
        xaxis=observables['fitness'][names].values,
        yaxis=observables['gdp'][names].values,
        names=[get_fixed_name(name) for name in observables['fitness'].columns if name in INCLUDE_CTRS_SET],
    )
    plot_country_trajectories(
        ax=ax_top,
        xaxis=observables['fitness'].values,
        yaxis=observables['gdp'].values,
        names=["" for _ in observables['fitness'].columns],
        traj_colors=False, traj_alpha=0.3,
    )
    
    ax_top.set_ylabel('$log_{10}$(GDP)')
    ax_top.set_xlabel('$log_{10}$(Fitness)')
    ax_top.set_xlim(global_xlim)
    ax_top.set_ylim(global_ylim)
    ax_top.set_title(f'Best model: {model0} (orange) vs {model1} (teal)', fontsize='medium')
    
    # Bottom panel: p-values
    pvalues_for_plot = np.copy(bootstrap_pvalues)
    min_pvalue = 1e-10
    pvalues_for_plot[pvalues_for_plot < min_pvalue] = min_pvalue
    pvalues_for_plot[pvalues_for_plot > 1.0] = 1.0
    
    pvalues_log10 = np.log10(pvalues_for_plot)
    center_value = bh_threshold
    center_log10 = np.log10(center_value)
    
    norm = TwoSlopeNorm(vmin=np.log10(min_pvalue), vcenter=center_log10, vmax=0)
    cmap = plt.cm.coolwarm
    
    for idx, (x, y, pval_log) in enumerate(zip(coords[0].flatten(), coords[1].flatten(), pvalues_log10)):
        if np.isnan(pval_log):
            color = 'grey'
            alpha = 0.3
        else:
            color = cmap(norm(pval_log))
            alpha = patches_alpha
        
        pixel = mpatches.Rectangle(
            xy=(x, y), width=xpixelsize, height=ypixelsize,
            fill=True, edgecolor='none', alpha=alpha,
            linewidth=0, facecolor=color
        )
        ax_bottom.add_patch(pixel)
    
    plot_country_trajectories(
        ax=ax_bottom,
        xaxis=observables['fitness'][names].values,
        yaxis=observables['gdp'][names].values,
        names=[get_fixed_name(name) for name in observables['fitness'].columns if name in INCLUDE_CTRS_SET],
    )
    plot_country_trajectories(
        ax=ax_bottom,
        xaxis=observables['fitness'].values,
        yaxis=observables['gdp'].values,
        names=["" for _ in observables['fitness'].columns],
        traj_colors=False, traj_alpha=0.3,
    )
    
    ax_bottom.set_ylabel('$log_{10}$(GDP)')
    ax_bottom.set_xlabel('$log_{10}$(Fitness)')
    ax_bottom.set_xlim(global_xlim)
    ax_bottom.set_ylim(global_ylim)
    
    # Colorbar
    divider = make_axes_locatable(ax_bottom)
    cbar_ax = divider.append_axes("right", size="15%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    
    def log_to_pval(x, pos):
        return f'{10**x:.0e}'
    
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(log_to_pval))
    cbar.set_label('Bootstrap p-value', rotation=270, labelpad=20)
    cbar.ax.axhline(y=center_log10, color='black', linestyle='--', linewidth=2)
    
    n_rejected = np.sum(reject_full)
    ax_bottom.set_title(
        f'Bootstrap p-values: {model0} vs {model1} ({n_bootstrap} replicates)\n'
        f'(blue = significant difference; threshold = {center_value:.2e})',
        fontsize='medium'
    )
    
    plt.tight_layout()
    
    figname = f'si/dt{DT}_combined_2models_bootstrap_{n_bootstrap}_2d2d'
    savefig(fig_combined, figname, output_dir)
    
    print(f"Significant differences: {n_rejected}/{len(valid_pvalues)} pixels")


def plot_bootstrap_3models(data, observables, output_dir, nperiods, DT,
                           n_bootstrap=500, n_jobs=4):
    """
    Bootstrap analysis comparing 3 models: gdp-polity, fitness-gdp, gdp-tech_fitness.
    Produces: dt{DT}_combined_3models_bootstrap_{n_bootstrap}_tf_2d2d2d.pdf
    """
    included_models = ['gdp-polity', 'fitness-gdp', 'gdp-tech_fitness']
    
    df = data.copy()
    df = take_only_last_n_periods_extended(df, nperiods)
    df = df.loc[df.pred_dt == DT, :]
    
    # Use same limits as other figures
    global_xlim = (-4, 1.5)  # fitness limits
    global_ylim = (2.7, 5)   # GDP limits
    
    _scale = 0.66
    patches_alpha = 0.3
    resolution = 30
    
    xpixelsize = (global_xlim[1] - global_xlim[0]) / resolution
    ypixelsize = (global_ylim[1] - global_ylim[0]) / resolution
    
    bwscale = 3.5
    bw = xpixelsize * bwscale, ypixelsize * bwscale
    
    X = df[['value_fitness', 'value_gdp']].values
    
    x0basepositions = np.linspace(*global_xlim, resolution)
    x1basepositions = np.linspace(*global_ylim, resolution)
    coords = np.meshgrid(x0basepositions, x1basepositions)
    nwkr_coords = np.stack([coords[0].flatten(), coords[1].flatten()]).T
    
    # Fit original models
    results_pred = {}
    for modelname in included_models:
        y = df[f'prediction_mae_vspsb_{modelname}'].values
        model = NWKR(bandwidth=bw)
        model.fit(X=X, y=y)
        _pred, _ = model.predict(nwkr_coords)
        results_pred[modelname] = _pred
    
    # Find which model is best at each pixel
    stacked = np.vstack([results_pred[m] for m in included_models])
    sorted_indices = np.argsort(stacked, axis=0)
    best_idx = sorted_indices[0, :]
    second_best_idx = sorted_indices[1, :]
    whichmin = best_idx
    colors = [MODELCOL[m] for m in included_models]
    
    print(f"Starting 3-model bootstrap with {n_bootstrap} replicates...")
    
    model_y_data = {m: df[f'prediction_mae_vspsb_{m}'].values for m in included_models}
    
    def bootstrap_replicate(b):
        np.random.seed(b)
        n = len(X)
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = X[idx]
        
        boot_pred = {}
        for m in included_models:
            y_boot = model_y_data[m][idx]
            model = NWKR(bandwidth=bw)
            model.fit(X=X_boot, y=y_boot)
            _pred, _ = model.predict(nwkr_coords)
            boot_pred[m] = _pred
        return boot_pred
    
    bootstrap_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(bootstrap_replicate)(b) for b in range(n_bootstrap)
    )
    
    print("Computing p-values...")
    n_pixels = len(nwkr_coords)
    bootstrap_pvalues = np.zeros(n_pixels)
    bootstrap_win_rate = np.zeros(n_pixels)
    
    for pixel_idx in range(n_pixels):
        best_model = included_models[best_idx[pixel_idx]]
        second_best_model = included_models[second_best_idx[pixel_idx]]
        
        best_boot = np.array([bootstrap_results[b][best_model][pixel_idx] for b in range(n_bootstrap)])
        second_boot = np.array([bootstrap_results[b][second_best_model][pixel_idx] for b in range(n_bootstrap)])
        
        valid = ~(np.isnan(best_boot) | np.isnan(second_boot))
        if np.sum(valid) == 0:
            bootstrap_pvalues[pixel_idx] = np.nan
            bootstrap_win_rate[pixel_idx] = np.nan
            continue
        
        best_boot = best_boot[valid]
        second_boot = second_boot[valid]
        
        best_wins = np.sum(best_boot < second_boot)
        bootstrap_win_rate[pixel_idx] = best_wins / len(best_boot)
        
        second_wins = np.sum(second_boot < best_boot)
        p_best = best_wins / len(best_boot)
        p_second = second_wins / len(best_boot)
        
        bootstrap_pvalues[pixel_idx] = min(2 * min(p_best, p_second), 1.0)
    
    # FDR correction
    valid_mask = ~np.isnan(bootstrap_pvalues)
    valid_pvalues = bootstrap_pvalues[valid_mask]
    
    reject, _, _, _ = multipletests(valid_pvalues, alpha=0.05, method='fdr_bh')
    bh_threshold = np.max(valid_pvalues[reject]) if np.any(reject) else 0.05
    
    reject_full = np.zeros(n_pixels, dtype=bool)
    reject_full[valid_mask] = reject
    
    # Create figure
    fig_combined, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(15 * _scale, 20 * _scale))
    
    # Top panel: which model wins
    for x, y, c in zip(coords[0].flatten(), coords[1].flatten(), whichmin):
        pixel = mpatches.Rectangle(
            xy=(x, y), width=xpixelsize, height=ypixelsize,
            fill=True, edgecolor='none', alpha=patches_alpha,
            linewidth=0, facecolor=colors[c]
        )
        ax_top.add_patch(pixel)
    
    names = [name for name in observables['fitness'].columns if name in INCLUDE_CTRS_SET]
    plot_country_trajectories(
        ax=ax_top,
        xaxis=observables['fitness'][names].values,
        yaxis=observables['gdp'][names].values,
        names=[get_fixed_name(name) for name in observables['fitness'].columns if name in INCLUDE_CTRS_SET],
    )
    plot_country_trajectories(
        ax=ax_top,
        xaxis=observables['fitness'].values,
        yaxis=observables['gdp'].values,
        names=["" for _ in observables['fitness'].columns],
        traj_colors=False, traj_alpha=0.3,
    )
    
    ax_top.set_ylabel(r'$log_{10}$(GDP)')
    ax_top.set_xlabel(r'$log_{10}$(Fitness)')
    ax_top.set_xlim(global_xlim)
    ax_top.set_ylim(global_ylim)
    ax_top.set_title('Best predictor model (3-model comparison)', fontsize='medium')
    
    # Legend for top plot
    divider_top = make_axes_locatable(ax_top)
    legend_ax = divider_top.append_axes("right", size="15%", pad=0.1)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(False)
    
    legend_dict = {m: MODELCOL[m] for m in included_models}
    num_items = len(legend_dict)
    spacing = 0.9 / (num_items + 1)
    legend_ax.set_title('Lowest error\nachieved by', fontsize='medium', y=0.90, x=1.1)
    
    for i, (name, color) in enumerate(legend_dict.items()):
        y_pos = 1 - 0.1 - spacing * (i + 1)
        patch = mpatches.Rectangle((0.1, y_pos), 0.2, spacing * 0.8, 
                                   facecolor=color, edgecolor=None, alpha=patches_alpha)
        legend_ax.add_patch(patch)
        legend_ax.text(0.35, y_pos + spacing * 0.4, name, va='center', ha='left')
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    
    # Bottom panel: p-values
    pvalues_for_plot = np.copy(bootstrap_pvalues)
    min_pvalue = 1e-10
    pvalues_for_plot[pvalues_for_plot < min_pvalue] = min_pvalue
    pvalues_for_plot[pvalues_for_plot > 1.0] = 1.0
    
    pvalues_log10 = np.log10(pvalues_for_plot)
    center_value = bh_threshold if bh_threshold > 0 else 0.05
    center_log10 = np.log10(center_value)
    
    norm = TwoSlopeNorm(vmin=np.log10(min_pvalue), vcenter=center_log10, vmax=0)
    cmap = plt.cm.coolwarm
    
    for idx, (x, y, pval_log) in enumerate(zip(coords[0].flatten(), coords[1].flatten(), pvalues_log10)):
        if np.isnan(pval_log):
            color = 'grey'
            alpha = 0.3
        else:
            color = cmap(norm(pval_log))
            alpha = patches_alpha
        
        pixel = mpatches.Rectangle(
            xy=(x, y), width=xpixelsize, height=ypixelsize,
            fill=True, edgecolor='none', alpha=alpha,
            linewidth=0, facecolor=color
        )
        ax_bottom.add_patch(pixel)
    
    plot_country_trajectories(
        ax=ax_bottom,
        xaxis=observables['fitness'][names].values,
        yaxis=observables['gdp'][names].values,
        names=[get_fixed_name(name) for name in observables['fitness'].columns if name in INCLUDE_CTRS_SET],
    )
    plot_country_trajectories(
        ax=ax_bottom,
        xaxis=observables['fitness'].values,
        yaxis=observables['gdp'].values,
        names=["" for _ in observables['fitness'].columns],
        traj_colors=False, traj_alpha=0.3,
    )
    
    ax_bottom.set_ylabel(r'$log_{10}$(GDP)')
    ax_bottom.set_xlabel(r'$log_{10}$(Fitness)')
    ax_bottom.set_xlim(global_xlim)
    ax_bottom.set_ylim(global_ylim)
    
    # Colorbar
    divider_bottom = make_axes_locatable(ax_bottom)
    cbar_ax = divider_bottom.append_axes("right", size="15%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    
    def log_to_pval(x, pos):
        return f'{10**x:.0e}'
    
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(log_to_pval))
    cbar.set_label('Bootstrap p-value', rotation=270, labelpad=20)
    cbar.ax.axhline(y=center_log10, color='black', linestyle='--', linewidth=2)
    
    n_rejected = np.sum(reject_full)
    ax_bottom.set_title(
        f'Bootstrap p-values: best vs second-best ({n_bootstrap} replicates)\n'
        f'(blue = significant difference; threshold = {center_value:.2e})',
        fontsize='medium'
    )
    
    plt.tight_layout()
    savefig(fig_combined, f'si/dt{DT}_combined_3models_bootstrap_{n_bootstrap}_tf_2d2d2d', output_dir)
    
    print(f"Significant differences: {n_rejected}/{len(valid_pvalues)} pixels")


def plot_bootstrap_bydim(data, observables, output_dir, nperiods, DT,
                         n_bootstrap=500, n_jobs=4):
    """
    Bootstrap analysis comparing models by dimensionality.
    Produces: dt{DT}_combined_bydim_bootstrap_{n_bootstrap}_tf_all.pdf
    """
    # All 7 models
    included_models = [
        'fitness-gdp-polity',
        'gdp-polity',
        'fitness-gdp',
        'fitness-gdp-polity-tech_fitness',
        'fitness-gdp-tech_fitness',
        'gdp-tech_fitness',
        'gdp-polity-tech_fitness',
    ]
    
    # Dimensionality mapping
    model_to_dim = {
        'fitness-gdp': '2D',
        'fitness-gdp-polity': '3D',
        'gdp-polity': '2D',
        'fitness-gdp-polity-tech_fitness': '4D',
        'fitness-gdp-tech_fitness': '3D',
        'gdp-tech_fitness': '2D',
        'gdp-polity-tech_fitness': '3D',
    }
    
    _4D = '#ff7043'
    _3D = '#0077bb'
    _2D = '#009988'
    
    dim_colors = {'4D': _4D, '3D': _3D, '2D': _2D}
    temp_modelcol = {m: dim_colors[model_to_dim[m]] for m in included_models}
    
    df = data.copy()
    df = take_only_last_n_periods_extended(df, nperiods)
    df = df.loc[df.pred_dt == DT, :]
    
    # Use same limits as other figures
    global_xlim = (-4, 1.5)  # fitness limits
    global_ylim = (2.7, 5)   # GDP limits
    
    _scale = 0.66
    patches_alpha = 0.3
    resolution = 30
    
    xpixelsize = (global_xlim[1] - global_xlim[0]) / resolution
    ypixelsize = (global_ylim[1] - global_ylim[0]) / resolution
    
    bwscale = 3
    bw = xpixelsize * bwscale, ypixelsize * bwscale
    
    X = df[['value_fitness', 'value_gdp']].values
    
    x0basepositions = np.linspace(*global_xlim, resolution)
    x1basepositions = np.linspace(*global_ylim, resolution)
    coords = np.meshgrid(x0basepositions, x1basepositions)
    nwkr_coords = np.stack([coords[0].flatten(), coords[1].flatten()]).T
    
    # Fit original models
    results_pred = {}
    for modelname in included_models:
        y = df[f'prediction_mae_vspsb_{modelname}'].values
        model = NWKR(bandwidth=bw)
        model.fit(X=X, y=y)
        _pred, _ = model.predict(nwkr_coords)
        results_pred[modelname] = _pred
    
    # For each pixel, find best model per dimensionality
    n_pixels = len(nwkr_coords)
    dimensionalities = ['4D', '3D', '2D']
    
    dim_best_error = {dim: np.full(n_pixels, np.inf) for dim in dimensionalities}
    dim_best_model = {dim: [None] * n_pixels for dim in dimensionalities}
    
    for pixel_idx in range(n_pixels):
        for model in included_models:
            dim = model_to_dim[model]
            error = results_pred[model][pixel_idx]
            if error < dim_best_error[dim][pixel_idx]:
                dim_best_error[dim][pixel_idx] = error
                dim_best_model[dim][pixel_idx] = model
    
    # Find best and second-best dimensionality at each pixel
    best_dim = []
    second_best_dim = []
    whichmin = []
    
    for pixel_idx in range(n_pixels):
        dim_errors = [(dim, dim_best_error[dim][pixel_idx]) for dim in dimensionalities]
        dim_errors.sort(key=lambda x: x[1])
        
        best_dim.append(dim_errors[0][0])
        second_best_dim.append(dim_errors[1][0])
        
        best_model = dim_best_model[dim_errors[0][0]][pixel_idx]
        whichmin.append(included_models.index(best_model))
    
    whichmin = np.array(whichmin)
    colors = [temp_modelcol[m] for m in included_models]
    
    print(f"Starting by-dimension bootstrap with {n_bootstrap} replicates...")
    
    model_y_data = {m: df[f'prediction_mae_vspsb_{m}'].values for m in included_models}
    
    def bootstrap_replicate(b):
        np.random.seed(b)
        n = len(X)
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = X[idx]
        
        boot_pred = {}
        for m in included_models:
            y_boot = model_y_data[m][idx]
            model = NWKR(bandwidth=bw)
            model.fit(X=X_boot, y=y_boot)
            _pred, _ = model.predict(nwkr_coords)
            boot_pred[m] = _pred
        return boot_pred
    
    bootstrap_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(bootstrap_replicate)(b) for b in range(n_bootstrap)
    )
    
    print("Computing p-values (comparing dimensionalities)...")
    bootstrap_pvalues = np.zeros(n_pixels)
    
    for pixel_idx in range(n_pixels):
        best_d = best_dim[pixel_idx]
        second_d = second_best_dim[pixel_idx]
        
        best_dim_errors = []
        second_dim_errors = []
        
        for b in range(n_bootstrap):
            # Best error for best_dim
            best_d_error = np.inf
            for model in included_models:
                if model_to_dim[model] == best_d:
                    error = bootstrap_results[b][model][pixel_idx]
                    if error < best_d_error:
                        best_d_error = error
            
            # Best error for second_dim
            second_d_error = np.inf
            for model in included_models:
                if model_to_dim[model] == second_d:
                    error = bootstrap_results[b][model][pixel_idx]
                    if error < second_d_error:
                        second_d_error = error
            
            best_dim_errors.append(best_d_error)
            second_dim_errors.append(second_d_error)
        
        best_dim_errors = np.array(best_dim_errors)
        second_dim_errors = np.array(second_dim_errors)
        
        valid = ~(np.isnan(best_dim_errors) | np.isnan(second_dim_errors) | 
                  np.isinf(best_dim_errors) | np.isinf(second_dim_errors))
        
        if np.sum(valid) == 0:
            bootstrap_pvalues[pixel_idx] = np.nan
            continue
        
        best_dim_errors = best_dim_errors[valid]
        second_dim_errors = second_dim_errors[valid]
        
        best_wins = np.sum(best_dim_errors < second_dim_errors)
        second_wins = np.sum(second_dim_errors < best_dim_errors)
        
        p_best = best_wins / len(best_dim_errors)
        p_second = second_wins / len(best_dim_errors)
        
        bootstrap_pvalues[pixel_idx] = min(2 * min(p_best, p_second), 1.0)
    
    # FDR correction
    valid_mask = ~np.isnan(bootstrap_pvalues)
    valid_pvalues = bootstrap_pvalues[valid_mask]
    
    reject, _, _, _ = multipletests(valid_pvalues, alpha=0.05, method='fdr_bh')
    bh_threshold = np.max(valid_pvalues[reject]) if np.any(reject) else 0.05
    
    reject_full = np.zeros(n_pixels, dtype=bool)
    reject_full[valid_mask] = reject
    
    # Create figure
    fig_combined, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(15 * _scale, 20 * _scale))
    
    # Top panel
    for x, y, c in zip(coords[0].flatten(), coords[1].flatten(), whichmin):
        pixel = mpatches.Rectangle(
            xy=(x, y), width=xpixelsize, height=ypixelsize,
            fill=True, edgecolor='none', alpha=patches_alpha,
            linewidth=0, facecolor=colors[c]
        )
        ax_top.add_patch(pixel)
    
    names = [name for name in observables['fitness'].columns if name in INCLUDE_CTRS_SET]
    plot_country_trajectories(
        ax=ax_top,
        xaxis=observables['fitness'][names].values,
        yaxis=observables['gdp'][names].values,
        names=[get_fixed_name(name) for name in observables['fitness'].columns if name in INCLUDE_CTRS_SET],
    )
    plot_country_trajectories(
        ax=ax_top,
        xaxis=observables['fitness'].values,
        yaxis=observables['gdp'].values,
        names=["" for _ in observables['fitness'].columns],
        traj_colors=False, traj_alpha=0.3,
    )
    
    ax_top.set_ylabel(r'$log_{10}$(GDP)')
    ax_top.set_xlabel(r'$log_{10}$(Fitness)')
    ax_top.set_xlim(global_xlim)
    ax_top.set_ylim(global_ylim)
    ax_top.set_title('Best dimensionality', fontsize='medium')
    
    # Legend
    divider_top = make_axes_locatable(ax_top)
    legend_ax = divider_top.append_axes("right", size="15%", pad=0.1)
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    for spine in legend_ax.spines.values():
        spine.set_visible(False)
    
    legend_items = [('4D', _4D), ('3D', _3D), ('2D', _2D)]
    num_items = len(legend_items)
    spacing = 0.9 / (num_items + 1)
    legend_ax.set_title('Lowest error\nmodel', fontsize='medium', y=0.90, x=1.1)
    
    for i, (name, color) in enumerate(legend_items):
        y_pos = 1 - 0.1 - spacing * (i + 1)
        patch = mpatches.Rectangle((0.1, y_pos), 0.2, spacing * 0.8, 
                                   facecolor=color, edgecolor=None, alpha=patches_alpha)
        legend_ax.add_patch(patch)
        legend_ax.text(0.35, y_pos + spacing * 0.4, name, va='center', ha='left')
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    
    # Bottom panel: p-values
    pvalues_for_plot = np.copy(bootstrap_pvalues)
    min_pvalue = 1e-10
    pvalues_for_plot[pvalues_for_plot < min_pvalue] = min_pvalue
    pvalues_for_plot[pvalues_for_plot > 1.0] = 1.0
    
    pvalues_log10 = np.log10(pvalues_for_plot)
    center_value = bh_threshold if bh_threshold > 0 else 0.05
    center_log10 = np.log10(center_value)
    
    norm = TwoSlopeNorm(vmin=np.log10(min_pvalue), vcenter=center_log10, vmax=0)
    cmap = plt.cm.coolwarm
    
    for idx, (x, y, pval_log) in enumerate(zip(coords[0].flatten(), coords[1].flatten(), pvalues_log10)):
        if np.isnan(pval_log):
            color = 'grey'
            alpha = 0.3
        else:
            color = cmap(norm(pval_log))
            alpha = patches_alpha
        
        pixel = mpatches.Rectangle(
            xy=(x, y), width=xpixelsize, height=ypixelsize,
            fill=True, edgecolor='none', alpha=alpha,
            linewidth=0, facecolor=color
        )
        ax_bottom.add_patch(pixel)
    
    plot_country_trajectories(
        ax=ax_bottom,
        xaxis=observables['fitness'][names].values,
        yaxis=observables['gdp'][names].values,
        names=[get_fixed_name(name) for name in observables['fitness'].columns if name in INCLUDE_CTRS_SET],
    )
    plot_country_trajectories(
        ax=ax_bottom,
        xaxis=observables['fitness'].values,
        yaxis=observables['gdp'].values,
        names=["" for _ in observables['fitness'].columns],
        traj_colors=False, traj_alpha=0.3,
    )
    
    ax_bottom.set_ylabel(r'$log_{10}$(GDP)')
    ax_bottom.set_xlabel(r'$log_{10}$(Fitness)')
    ax_bottom.set_xlim(global_xlim)
    ax_bottom.set_ylim(global_ylim)
    
    # Colorbar
    divider_bottom = make_axes_locatable(ax_bottom)
    cbar_ax = divider_bottom.append_axes("right", size="15%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    
    def log_to_pval(x, pos):
        return f'{10**x:.0e}'
    
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(log_to_pval))
    cbar.set_label('Bootstrap p-value', rotation=270, labelpad=20)
    cbar.ax.axhline(y=center_log10, color='black', linestyle='--', linewidth=2)
    
    n_rejected = np.sum(reject_full)
    ax_bottom.set_title(
        f'Bootstrap p-values: best vs second-best dimensionality ({n_bootstrap} replicates)\n'
        f'(blue = significant difference; threshold = {center_value:.2e})',
        fontsize='medium'
    )
    
    plt.tight_layout()
    savefig(fig_combined, f'si/dt{DT}_combined_bydim_bootstrap_{n_bootstrap}_tf_all', output_dir)
    
    print(f"Significant differences: {n_rejected}/{len(valid_pvalues)} pixels")


def plot_bootstrap_4panel(data, observables, output_dir, nperiods, DT,
                          n_bootstrap=500, n_jobs=4):
    """
    Four-panel bootstrap comparison of dimensionalities.
    Produces: dt{DT}_4panel_dimensionality_comparison_{n_bootstrap}.pdf
    
    Panels:
    (a) Best vs second-best dimensionality
    (b) 2D vs 3D models
    (c) 2D vs 4D models
    (d) 3D vs 4D models
    """
    included_models = [
        'fitness-gdp-polity',
        'gdp-polity',
        'fitness-gdp',
        'fitness-gdp-polity-tech_fitness',
        'fitness-gdp-tech_fitness',
        'gdp-tech_fitness',
        'gdp-polity-tech_fitness',
    ]
    
    model_to_dim = {
        'fitness-gdp': '2D',
        'fitness-gdp-polity': '3D',
        'gdp-polity': '2D',
        'fitness-gdp-polity-tech_fitness': '4D',
        'fitness-gdp-tech_fitness': '3D',
        'gdp-tech_fitness': '2D',
        'gdp-polity-tech_fitness': '3D',
    }
    
    _4D = '#ff7043'
    _3D = '#0077bb'
    _2D = '#009988'
    
    df = data.copy()
    df = take_only_last_n_periods_extended(df, nperiods)
    df = df.loc[df.pred_dt == DT, :]
    
    # Use same limits as other figures
    global_xlim = (-4, 1.5)  # fitness limits
    global_ylim = (2.7, 5)   # GDP limits
    
    _scale = 0.66
    patches_alpha = 0.3
    resolution = 30
    
    xpixelsize = (global_xlim[1] - global_xlim[0]) / resolution
    ypixelsize = (global_ylim[1] - global_ylim[0]) / resolution
    
    bwscale = 3
    bw = xpixelsize * bwscale, ypixelsize * bwscale
    
    X = df[['value_fitness', 'value_gdp']].values
    
    x0basepositions = np.linspace(*global_xlim, resolution)
    x1basepositions = np.linspace(*global_ylim, resolution)
    coords = np.meshgrid(x0basepositions, x1basepositions)
    nwkr_coords = np.stack([coords[0].flatten(), coords[1].flatten()]).T
    n_pixels = len(nwkr_coords)
    
    # Fit original models
    results_pred = {}
    for modelname in included_models:
        y = df[f'prediction_mae_vspsb_{modelname}'].values
        model = NWKR(bandwidth=bw)
        model.fit(X=X, y=y)
        _pred, _ = model.predict(nwkr_coords)
        results_pred[modelname] = _pred
    
    print(f"Starting 4-panel bootstrap with {n_bootstrap} replicates...")
    
    model_y_data = {m: df[f'prediction_mae_vspsb_{m}'].values for m in included_models}
    
    def bootstrap_replicate(b):
        np.random.seed(b)
        n = len(X)
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = X[idx]
        
        boot_pred = {}
        for m in included_models:
            y_boot = model_y_data[m][idx]
            model = NWKR(bandwidth=bw)
            model.fit(X=X_boot, y=y_boot)
            _pred, _ = model.predict(nwkr_coords)
            boot_pred[m] = _pred
        return boot_pred
    
    bootstrap_results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(bootstrap_replicate)(b) for b in range(n_bootstrap)
    )
    
    def get_best_dim_error(bootstrap_results, pixel_idx, dim, b):
        """Get the best error for a given dimensionality at a pixel."""
        best_error = np.inf
        for model in included_models:
            if model_to_dim[model] == dim:
                error = bootstrap_results[b][model][pixel_idx]
                if error < best_error:
                    best_error = error
        return best_error
    
    def compute_pairwise_pvalues(dim1, dim2):
        """Compute p-values comparing two dimensionalities."""
        pvalues = np.zeros(n_pixels)
        which_wins = np.zeros(n_pixels)  # 0 for dim1, 1 for dim2
        
        for pixel_idx in range(n_pixels):
            dim1_errors = [get_best_dim_error(bootstrap_results, pixel_idx, dim1, b) 
                          for b in range(n_bootstrap)]
            dim2_errors = [get_best_dim_error(bootstrap_results, pixel_idx, dim2, b) 
                          for b in range(n_bootstrap)]
            
            dim1_errors = np.array(dim1_errors)
            dim2_errors = np.array(dim2_errors)
            
            valid = ~(np.isnan(dim1_errors) | np.isnan(dim2_errors) | 
                      np.isinf(dim1_errors) | np.isinf(dim2_errors))
            
            if np.sum(valid) == 0:
                pvalues[pixel_idx] = np.nan
                which_wins[pixel_idx] = np.nan
                continue
            
            dim1_errors = dim1_errors[valid]
            dim2_errors = dim2_errors[valid]
            
            # Determine which wins in original data
            dim1_orig = np.inf
            dim2_orig = np.inf
            for model in included_models:
                if model_to_dim[model] == dim1:
                    if results_pred[model][pixel_idx] < dim1_orig:
                        dim1_orig = results_pred[model][pixel_idx]
                elif model_to_dim[model] == dim2:
                    if results_pred[model][pixel_idx] < dim2_orig:
                        dim2_orig = results_pred[model][pixel_idx]
            
            which_wins[pixel_idx] = 0 if dim1_orig < dim2_orig else 1
            
            dim1_wins = np.sum(dim1_errors < dim2_errors)
            dim2_wins = np.sum(dim2_errors < dim1_errors)
            
            p_dim1 = dim1_wins / len(dim1_errors)
            p_dim2 = dim2_wins / len(dim1_errors)
            
            pvalues[pixel_idx] = min(2 * min(p_dim1, p_dim2), 1.0)
        
        return pvalues, which_wins
    
    print("Computing pairwise p-values...")
    
    # Panel (a): best vs second-best (use bydim result)
    dimensionalities = ['4D', '3D', '2D']
    dim_best_error = {dim: np.full(n_pixels, np.inf) for dim in dimensionalities}
    
    for pixel_idx in range(n_pixels):
        for model in included_models:
            dim = model_to_dim[model]
            error = results_pred[model][pixel_idx]
            if error < dim_best_error[dim][pixel_idx]:
                dim_best_error[dim][pixel_idx] = error
    
    best_dim = []
    second_best_dim = []
    for pixel_idx in range(n_pixels):
        dim_errors = [(dim, dim_best_error[dim][pixel_idx]) for dim in dimensionalities]
        dim_errors.sort(key=lambda x: x[1])
        best_dim.append(dim_errors[0][0])
        second_best_dim.append(dim_errors[1][0])
    
    # Compute p-values for panel (a)
    pvalues_a = np.zeros(n_pixels)
    which_wins_a = np.zeros(n_pixels)
    
    for pixel_idx in range(n_pixels):
        bd = best_dim[pixel_idx]
        sd = second_best_dim[pixel_idx]
        
        best_errors = [get_best_dim_error(bootstrap_results, pixel_idx, bd, b) 
                      for b in range(n_bootstrap)]
        second_errors = [get_best_dim_error(bootstrap_results, pixel_idx, sd, b) 
                        for b in range(n_bootstrap)]
        
        best_errors = np.array(best_errors)
        second_errors = np.array(second_errors)
        
        valid = ~(np.isnan(best_errors) | np.isnan(second_errors) | 
                  np.isinf(best_errors) | np.isinf(second_errors))
        
        if np.sum(valid) == 0:
            pvalues_a[pixel_idx] = np.nan
            which_wins_a[pixel_idx] = dimensionalities.index(bd)
            continue
        
        best_errors = best_errors[valid]
        second_errors = second_errors[valid]
        
        which_wins_a[pixel_idx] = dimensionalities.index(bd)
        
        best_wins = np.sum(best_errors < second_errors)
        second_wins = np.sum(second_errors < best_errors)
        
        p_best = best_wins / len(best_errors)
        p_second = second_wins / len(best_errors)
        
        pvalues_a[pixel_idx] = min(2 * min(p_best, p_second), 1.0)
    
    # Pairwise comparisons
    pvalues_2v3, which_wins_2v3 = compute_pairwise_pvalues('2D', '3D')
    pvalues_2v4, which_wins_2v4 = compute_pairwise_pvalues('2D', '4D')
    pvalues_3v4, which_wins_3v4 = compute_pairwise_pvalues('3D', '4D')
    
    # Find global BH threshold from all panels
    all_pvalues = np.concatenate([pvalues_a, pvalues_2v3, pvalues_2v4, pvalues_3v4])
    all_valid = ~np.isnan(all_pvalues)
    all_valid_pvalues = all_pvalues[all_valid]
    
    reject, _, _, _ = multipletests(all_valid_pvalues, alpha=0.05, method='fdr_bh')
    bh_threshold = np.max(all_valid_pvalues[reject]) if np.any(reject) else 0.05
    
    # Create 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(20 * _scale, 20 * _scale))
    
    panels = [
        (axes[0, 0], pvalues_a, which_wins_a, ['4D', '3D', '2D'], '(a) Best vs second-best dimensionality'),
        (axes[0, 1], pvalues_2v3, which_wins_2v3, ['2D', '3D'], '(b) 2D vs 3D'),
        (axes[1, 0], pvalues_2v4, which_wins_2v4, ['2D', '4D'], '(c) 2D vs 4D'),
        (axes[1, 1], pvalues_3v4, which_wins_3v4, ['3D', '4D'], '(d) 3D vs 4D'),
    ]
    
    dim_colors = {'4D': _4D, '3D': _3D, '2D': _2D}
    
    min_pvalue = 1e-10
    center_value = bh_threshold if bh_threshold > 0 else 0.05
    center_log10 = np.log10(center_value)
    norm = TwoSlopeNorm(vmin=np.log10(min_pvalue), vcenter=center_log10, vmax=0)
    cmap = plt.cm.coolwarm
    
    for ax, pvalues, which_wins, dims, title in panels:
        pvalues_for_plot = np.copy(pvalues)
        pvalues_for_plot[pvalues_for_plot < min_pvalue] = min_pvalue
        pvalues_for_plot[pvalues_for_plot > 1.0] = 1.0
        pvalues_log10 = np.log10(pvalues_for_plot)
        
        for idx, (x, y, pval_log) in enumerate(zip(coords[0].flatten(), coords[1].flatten(), pvalues_log10)):
            if np.isnan(pval_log):
                color = 'grey'
                alpha = 0.3
            else:
                color = cmap(norm(pval_log))
                alpha = patches_alpha
            
            pixel = mpatches.Rectangle(
                xy=(x, y), width=xpixelsize, height=ypixelsize,
                fill=True, edgecolor='none', alpha=alpha,
                linewidth=0, facecolor=color
            )
            ax.add_patch(pixel)
        
        # Add trajectories
        names = [name for name in observables['fitness'].columns if name in INCLUDE_CTRS_SET]
        plot_country_trajectories(
            ax=ax,
            xaxis=observables['fitness'][names].values,
            yaxis=observables['gdp'][names].values,
            names=[get_fixed_name(name) for name in observables['fitness'].columns if name in INCLUDE_CTRS_SET],
        )
        plot_country_trajectories(
            ax=ax,
            xaxis=observables['fitness'].values,
            yaxis=observables['gdp'].values,
            names=["" for _ in observables['fitness'].columns],
            traj_colors=False, traj_alpha=0.3,
        )
        
        ax.set_ylabel(r'$log_{10}$(GDP)')
        ax.set_xlabel(r'$log_{10}$(Fitness)')
        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)
        ax.set_title(title, fontsize='medium')
    
    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    
    def log_to_pval(x, pos):
        return f'{10**x:.0e}'
    
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(log_to_pval))
    cbar.set_label('Bootstrap p-value', rotation=270, labelpad=20)
    cbar.ax.axhline(y=center_log10, color='black', linestyle='--', linewidth=2)
    
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    savefig(fig, f'si/dt{DT}_4panel_dimensionality_comparison_{n_bootstrap}', output_dir)
    
    print("4-panel comparison complete!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate statistical analysis plots')
    parser.add_argument('--bootstrap', type=int, default=500,
                        help='Number of bootstrap replicates (default: 500)')
    parser.add_argument('--jobs', type=int, default=4,
                        help='Number of parallel jobs (default: 4)')
    parser.add_argument('--skip-bootstrap', action='store_true',
                        help='Skip bootstrap figures (faster)')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
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
    
    models = [x.replace('prediction_', '') for x in data.columns 
              if 'prediction' in x and 'vspsb' not in x]
    
    data = make_fair_comparison(data, models)
    data = add_country_column(data)
    compute_prediction_errors(data, models)
    compute_signed_errors(data, models)
    data = rename_mae_vspsb(data)
    
    observables = get_observables_dict_from_panel(data)
    
    nperiods = 5
    DT = 4
    
    print("\nGenerating statistical analysis plots...")
    
    # SPSb explanation figure (fast)
    print("\n=== SPSb Explanation Figure ===")
    plot_spsb_explanation(data, output_dir, seed=41)
    
    if not args.skip_bootstrap:
        print(f"\n=== Bootstrap Analysis (n_bootstrap={args.bootstrap}) ===")
        print("WARNING: This can take 30+ minutes per figure with n_bootstrap=500")
        
        # 2 models: gdp-polity vs fitness-gdp (SI)
        print("\n--- 2-model comparison (SI) ---")
        plot_bootstrap_2models(data, observables, output_dir, nperiods, DT,
                              'gdp-polity', 'fitness-gdp',
                              n_bootstrap=args.bootstrap, n_jobs=args.jobs)
        
        # 3 models: gdp-polity, fitness-gdp, gdp-tech_fitness (Main)
        print("\n--- 3-model comparison (Main) ---")
        plot_bootstrap_3models(data, observables, output_dir, nperiods, DT,
                              n_bootstrap=args.bootstrap, n_jobs=args.jobs)
        
        # By dimensionality (Main)
        print("\n--- By-dimension comparison (Main) ---")
        plot_bootstrap_bydim(data, observables, output_dir, nperiods, DT,
                            n_bootstrap=args.bootstrap, n_jobs=args.jobs)
        
        # 4-panel comparison (Main)
        print("\n--- 4-panel dimensionality comparison (Main) ---")
        plot_bootstrap_4panel(data, observables, output_dir, nperiods, DT,
                             n_bootstrap=args.bootstrap, n_jobs=args.jobs)
    else:
        print("\nSkipped bootstrap analysis (use --skip-bootstrap=false to enable)")
    
    print(f"\nAll statistical plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
