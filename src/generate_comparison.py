#!/usr/bin/env python
"""Generate comparison artefacts across TechFit imputation strategies."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from ectools.NWKR import NWKR

from plotting import (
    make_fair_comparison,
    add_country_column,
    compute_prediction_errors,
    compute_signed_errors,
    get_observables_dict_from_panel,
)

STRATEGIES = ['baseline', 'drop', 'country_min']

STRATEGY_LABELS = {
    'baseline': 'Baseline\n(global min)',
    'drop': 'Drop\n(remove NaN)',
    'country_min': 'Country min\n(backfill)',
}

# 3-model comparison: same models as the main-text figure
COMPARISON_MODELS = ['gdp-polity', 'fitness-gdp', 'gdp-tech_fitness']

MODELCOL = {
    'gdp-polity': '#009988',
    'fitness-gdp': '#ff7043',
    'gdp-tech_fitness': '#A0B710',
}

GLOBAL_XLIM = (-4, 1.5)
GLOBAL_YLIM = (2.7, 5)


def rename_mae_vspsb(df):
    """Rename columns for consistency."""
    df.columns = [x.replace('vspsb_mae', 'mae_vspsb') for x in df.columns]
    df.columns = [x.replace('vspsb_ae', 'ae_vspsb') for x in df.columns]
    return df


def take_only_last_n_periods_extended(df, n_periods=5):
    """Extended version that handles varying dt values."""
    max_nyears = df['pred_dt'].max()
    last_year = df['year_pred_end'].max()
    last_n_years = set(list(last_year - x for x in range(max_nyears + n_periods)))
    trimmed = df.loc[df['year_pred_end'].isin(last_n_years), :]
    return trimmed


def load_and_process(predictions_csv):
    """Load predictions CSV and compute errors."""
    data = pd.read_csv(predictions_csv)
    models = [
        x.replace('prediction_', '')
        for x in data.columns
        if 'prediction' in x and 'vspsb' not in x
    ]
    data = make_fair_comparison(data, models)
    data = add_country_column(data)
    compute_prediction_errors(data, models)
    compute_signed_errors(data, models)
    data = rename_mae_vspsb(data)
    return data, models


def generate_mae_summary(strategy_data, output_dir):
    """Generate MAE summary table across strategies.

    Args:
        strategy_data: dict mapping strategy name to (data, models) tuples
        output_dir: Path to write output files
    """
    nperiods = 5
    DT = 4
    rows = []
    for strategy in STRATEGIES:
        data, models = strategy_data[strategy]
        df = take_only_last_n_periods_extended(data, nperiods)
        df = df.loc[df.pred_dt == DT, :]
        for model in models:
            col = f'prediction_mae_vspsb_{model}'
            if col in df.columns:
                mae = df[col].mean()
                rows.append({'strategy': strategy, 'model': model, 'mae': mae})

    summary = pd.DataFrame(rows)
    pivot = summary.pivot(index='model', columns='strategy', values='mae')
    pivot = pivot[STRATEGIES]  # enforce column order
    pivot.to_csv(output_dir / 'mae_summary.csv')

    # LaTeX version (manual to avoid jinja2 dependency)
    with open(output_dir / 'mae_summary.tex', 'w') as f:
        f.write('\\begin{tabular}{l' + 'r' * len(STRATEGIES) + '}\n')
        f.write('\\toprule\n')
        f.write('model & ' + ' & '.join(STRATEGIES) + ' \\\\\n')
        f.write('\\midrule\n')
        for model in pivot.index:
            vals = ' & '.join(f'{pivot.loc[model, s]:.4f}' for s in STRATEGIES)
            f.write(f'{model} & {vals} \\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
    print(f"  MAE summary: {output_dir / 'mae_summary.csv'}")


def generate_data_summary(strategy_data, output_dir):
    """Generate dataset size summary across strategies."""
    rows = []
    for strategy in STRATEGIES:
        data, _ = strategy_data[strategy]
        df_dt4 = data.loc[data.pred_dt == 4, :]
        rows.append({
            'strategy': strategy,
            'total_obs': len(data),
            'obs_dt4': len(df_dt4),
            'n_countries': data['country_code'].nunique(),
            'year_min': int(data['year_pred_start'].min()),
            'year_max': int(data['year_pred_end'].max()),
        })
    summary = pd.DataFrame(rows).set_index('strategy')
    summary.to_csv(output_dir / 'data_summary.csv')
    print(f"  Data summary: {output_dir / 'data_summary.csv'}")
    print(summary.to_string())


def generate_whichbest_comparison(strategy_data, output_dir):
    """Generate side-by-side whichbest maps for three strategies."""
    nperiods = 5
    DT = 4
    resolution = 30
    patches_alpha = 0.3

    xpixelsize = (GLOBAL_XLIM[1] - GLOBAL_XLIM[0]) / resolution
    ypixelsize = (GLOBAL_YLIM[1] - GLOBAL_YLIM[0]) / resolution
    bwscale = 3
    bw = xpixelsize * bwscale, ypixelsize * bwscale

    x0 = np.linspace(*GLOBAL_XLIM, resolution)
    x1 = np.linspace(*GLOBAL_YLIM, resolution)
    coords = np.meshgrid(x0, x1)
    nwkr_coords = np.stack([coords[0].flatten(), coords[1].flatten()]).T

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, strategy in zip(axes, STRATEGIES):
        data, _ = strategy_data[strategy]
        df = take_only_last_n_periods_extended(data, nperiods)
        df = df.loc[df.pred_dt == DT, :]
        X = df[['value_fitness', 'value_gdp']].values

        results_pred = {}
        for model in COMPARISON_MODELS:
            col = f'prediction_mae_vspsb_{model}'
            if col not in df.columns:
                continue
            y = df[col].values
            nwkr = NWKR(bandwidth=bw)
            nwkr.fit(X=X, y=y)
            pred, _ = nwkr.predict(nwkr_coords)
            results_pred[model] = pred

        available = [m for m in COMPARISON_MODELS if m in results_pred]
        stacked = np.vstack([results_pred[m] for m in available])
        whichmin = np.argmin(stacked, axis=0)
        colors = [MODELCOL[m] for m in available]

        for x, y, c in zip(coords[0].flatten(), coords[1].flatten(), whichmin):
            pixel = mpatches.Rectangle(
                xy=(x, y), width=xpixelsize, height=ypixelsize,
                fill=True, edgecolor='none', alpha=patches_alpha,
                linewidth=0, facecolor=colors[c],
            )
            ax.add_patch(pixel)

        # Plot country-year dots
        ax.scatter(X[:, 0], X[:, 1], c='black', s=1, alpha=0.15, zorder=2)

        ax.set_xlim(*GLOBAL_XLIM)
        ax.set_ylim(*GLOBAL_YLIM)
        ax.set_xlabel(r'$\log_{10}$(Fitness)')
        ax.set_title(STRATEGY_LABELS[strategy])

    axes[0].set_ylabel(r'$\log_{10}$(GDP)')

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=MODELCOL[m], alpha=patches_alpha, label=m)
        for m in COMPARISON_MODELS
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               title='Lowest MAE achieved by', frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = output_dir / 'whichbest_comparison.pdf'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Whichbest comparison: {out}")


def _fit_nwkr_argmins(data, nperiods, DT, resolution, bw, nwkr_coords):
    """Fit NWKR for each comparison model and return per-cell argmin indices."""
    df = take_only_last_n_periods_extended(data, nperiods)
    df = df.loc[df.pred_dt == DT, :]
    X = df[['value_fitness', 'value_gdp']].values

    preds = {}
    for model in COMPARISON_MODELS:
        col = f'prediction_mae_vspsb_{model}'
        nwkr = NWKR(bandwidth=bw)
        nwkr.fit(X=X, y=df[col].values)
        pred, _ = nwkr.predict(nwkr_coords)
        preds[model] = pred

    stacked = np.vstack([preds[m] for m in COMPARISON_MODELS])
    return np.argmin(stacked, axis=0), X


def generate_stability_map(strategy_data, output_dir):
    """Generate a single-panel stability map showing where model rankings change.

    Shows the baseline whichbest map with hatching on cells where the drop
    strategy produces a different argmin, plus markers for dropped observations.
    """
    from matplotlib.lines import Line2D

    nperiods = 5
    DT = 4
    resolution = 30
    patches_alpha = 0.3

    # Extended y-range to include MOZ observations (GDP ~2.5)
    ylim = (2.4, GLOBAL_YLIM[1])

    xpixelsize = (GLOBAL_XLIM[1] - GLOBAL_XLIM[0]) / resolution
    ypixelsize = (ylim[1] - ylim[0]) / resolution
    bwscale = 3
    bw = xpixelsize * bwscale, ypixelsize * bwscale

    x0 = np.linspace(*GLOBAL_XLIM, resolution)
    x1 = np.linspace(*ylim, resolution)
    coords = np.meshgrid(x0, x1)
    nwkr_coords = np.stack([coords[0].flatten(), coords[1].flatten()]).T

    # Compute argmins for baseline and drop
    baseline_data, _ = strategy_data['baseline']
    drop_data, _ = strategy_data['drop']

    argmin_bl, X_bl = _fit_nwkr_argmins(baseline_data, nperiods, DT, resolution, bw, nwkr_coords)
    argmin_dr, X_dr = _fit_nwkr_argmins(drop_data, nperiods, DT, resolution, bw, nwkr_coords)

    flips = argmin_bl != argmin_dr
    colors_bl = [MODELCOL[COMPARISON_MODELS[i]] for i in argmin_bl]
    colors_dr = [MODELCOL[COMPARISON_MODELS[i]] for i in argmin_dr]

    # Identify dropped observations (in baseline but absent from drop, at dt=4)
    key = ['year_pred_start', 'year_pred_end', 'country_code', 'pred_dt']
    bl_dt4 = take_only_last_n_periods_extended(baseline_data, nperiods)
    bl_dt4 = bl_dt4.loc[bl_dt4.pred_dt == DT]
    dr_dt4 = take_only_last_n_periods_extended(drop_data, nperiods)
    dr_dt4 = dr_dt4.loc[dr_dt4.pred_dt == DT]

    merged = bl_dt4.merge(dr_dt4[key], on=key, how='left', indicator=True)
    dropped = merged[merged['_merge'] == 'left_only']

    _scale = 0.66
    fig, ax = plt.subplots(figsize=(15 * _scale, 10 * _scale))

    # Layer 1: baseline whichbest cells
    for x, y, c_bl, c_dr, is_flip in zip(
        coords[0].flatten(), coords[1].flatten(), colors_bl, colors_dr, flips,
    ):
        pixel = mpatches.Rectangle(
            xy=(x, y), width=xpixelsize, height=ypixelsize,
            fill=True, edgecolor='none', alpha=patches_alpha,
            linewidth=0, facecolor=c_bl,
        )
        ax.add_patch(pixel)

        # Layer 2: hatching on flip cells
        if is_flip:
            hatch_pixel = mpatches.Rectangle(
                xy=(x, y), width=xpixelsize, height=ypixelsize,
                fill=False, edgecolor=c_dr, linewidth=0.5,
                hatch='///', alpha=0.7,
            )
            ax.add_patch(hatch_pixel)

    # Layer 3: all observations (grey dots)
    ax.scatter(X_bl[:, 0], X_bl[:, 1], c='black', s=1, alpha=0.15, zorder=2)

    # Layer 4: dropped observations (red-edged circles with labels)
    if len(dropped) > 0:
        ax.scatter(
            dropped['value_fitness'].values,
            dropped['value_gdp'].values,
            s=40, facecolors='none', edgecolors='red', linewidths=1.5, zorder=4,
        )
        # Label unique country positions
        labelled = set()
        for _, row in dropped.iterrows():
            cc = row['country_code']
            if cc not in labelled:
                ax.annotate(
                    cc, (row['value_fitness'], row['value_gdp']),
                    textcoords='offset points', xytext=(5, 5),
                    fontsize=7, color='red', fontweight='bold', zorder=5,
                )
                labelled.add(cc)

    ax.set_xlim(*GLOBAL_XLIM)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r'$\log_{10}$(Fitness)')
    ax.set_ylabel(r'$\log_{10}$(GDP)')
    ax.set_title('Sensitivity of best-model ranking to TechFit imputation')

    # Legend
    legend_handles = [
        mpatches.Patch(color=MODELCOL[m], alpha=patches_alpha, label=m)
        for m in COMPARISON_MODELS
    ]
    legend_handles.append(
        mpatches.Patch(facecolor='white', edgecolor='grey', hatch='///',
                       label='Ranking changes\nunder drop strategy')
    )
    legend_handles.append(
        Line2D([0], [0], marker='o', color='w', markeredgecolor='red',
               markerfacecolor='none', markersize=8, linewidth=0,
               label='Dropped by\ndrop strategy')
    )
    ax.legend(handles=legend_handles, loc='upper left', fontsize='small',
              framealpha=0.9)

    n_flips = flips.sum()
    n_total = len(flips)
    ax.text(
        0.98, 0.02,
        f'{n_flips}/{n_total} cells change ({100*n_flips/n_total:.1f}%)\n'
        f'country_min: 0 changes',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=8, fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
    )

    fig.tight_layout()
    out = output_dir / 'stability_map.pdf'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Stability map: {out}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate comparison artefacts across TechFit imputation strategies')
    parser.add_argument('--base-output-dir', type=Path,
                        default=Path(__file__).parent / '../output',
                        help='Parent directory containing sensitivity_* dirs')
    args = parser.parse_args()

    base = args.base_output_dir
    comparison_dir = base / 'sensitivity_comparison'
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Load all strategies
    recompute_dir = comparison_dir / 'recompute_plots_with_other_strategies'
    strategy_data = {}
    for strategy in STRATEGIES:
        csv_path = recompute_dir / strategy / 'predictions' / 'polity-short-4d-backfill.csv'
        assert csv_path.exists(), f"Missing predictions for strategy '{strategy}': {csv_path}"
        print(f"Loading {strategy} from {csv_path}")
        strategy_data[strategy] = load_and_process(csv_path)

    print("\nGenerating comparison artefacts...")
    generate_data_summary(strategy_data, comparison_dir)
    generate_mae_summary(strategy_data, comparison_dir)
    generate_whichbest_comparison(strategy_data, comparison_dir)
    generate_stability_map(strategy_data, comparison_dir)

    print(f"\nAll comparison artefacts saved to: {comparison_dir}")


if __name__ == '__main__':
    main()
