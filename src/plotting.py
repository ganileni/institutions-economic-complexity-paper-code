import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from pathlib import Path
from collections import OrderedDict, defaultdict
from tqdm import tqdm
from scipy.stats import binom
from types import SimpleNamespace
import seaborn as sns
import pycountry
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from scipy.stats import norm
import warnings
import matplotlib
from ectools.NWKR import NWKR
from dieboldmariano import dm_test
import matplotlib.patheffects as pe


# https://github.com/tensorflow/tensorboard/blob/29cfcec9d15c47d1bf54b15df24e65bc38c188e6/tensorboard/components/tf_color_scale/palettes.ts
palette = OrderedDict({
    'orange':'#ff7043',
    'blue':'#0077bb',
    'teal':'#009988',
    'red':'#cc3311',
    'cyan':'#33bbee',
    'magenta':'#ee3377',
    'grey':'#bbbbbb',
})
# Set the default color cycle
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[item for key,item in palette.items()]) 

def country_code_from_name_dict(df):
    dictionary = dict()
    for code,name in zip(df.country_code,df.country):
        dictionary[name]=code
    return dictionary

def format_plot(ax,grid=True):
    if grid: ax.grid(which='major',alpha=.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

def add_xticklines(ax,**kwargs):
    for x in ax.get_xticks():
        ax.axvline(x,**kwargs)

def subset_countries(df,subdivision):
    return df.loc[df.country.isin(subdivision)]

def make_fair_comparison(df,models):
    all_prediction = df[[f'prediction_{model}' for model in models]]
    has_all_predictions = all_prediction.isnull().values.sum(axis=1) == 0
    # remove portion of dataset that does not have all the predictions for all the models
    df = df.loc[has_all_predictions,:]
    return df


def add_autocor_baseline(df):
    try:
        df['autocorrelation_baseline_mae'] = (df['last_observed_velocity']-df['groundtruth']).abs()
        df['autocorrelation_baseline_ae'] = df['last_observed_velocity']-df['groundtruth']
    except KeyError:
            print('Warning: can\'t add autocorrelation baseline because last_observed_velocity or groundtruth are missing' )

def compute_prediction_errors(df,models):
    for model in models:
        for _pred in ['prediction_vspsb','prediction']:
            df[f'{_pred}_mae_{model}']=(df[f'{_pred}_{model}']-df['groundtruth']).abs()
        add_autocor_baseline(df)

def compute_signed_errors(df,models):
    for model in models:
        for _pred in ['prediction_vspsb','prediction']:
            df[f'{_pred}_ae_{model}']=(df[f'{_pred}_{model}']-df['groundtruth'])
        add_autocor_baseline(df)
    
def lookup_country_name(country_code):
    return pycountry.countries.lookup(country_code).name

def add_country_column(df):
    countries = df['country_code'].unique()
    countries = {c:lookup_country_name(c) for c in countries}
    country_names = [countries[c] for c in df['country_code']]
    df['country'] = country_names
    return df


# take only N last periods, throw away rest
def take_only_last_n_periods(df,n_periods=3):
    result = list()
    for dt,_partial in df.groupby('pred_dt'):
        last_year = _partial['year_pred_end'].max()
        last_n_years = set(list(last_year - x for x in range(n_periods)))
        _partial = _partial.loc[_partial['year_pred_end'].isin(last_n_years),:]
        result.append(_partial)
    return pd.concat(result)

def drop_all_missing_last_velocities(df):
    _keep = df[[x for x in df.columns if 'prediction_vspsb' in x]]
    _keep = _keep.isnull().sum(axis=1)==0
    df = df.loc[_keep,:]
    return df

def bootstrap_df(df,dt=5):
    df = df.select_dtypes(include=np.number)
    df = df.loc[df.pred_dt==dt,:]
    df=drop_all_missing_last_velocities(df)
    results = list()
    for i in range(df.shape[0]):
        rnd = df.sample(replace=True,frac=1.)
        results.append(rnd.mean())
    results = pd.DataFrame(results)
    return results

def diebold_mariano(err1,err2,prob=.5,two_tailed=False,geq=False):
    "to evaluate hypothesis that model1 is consistently better than model2"
    # kill NaNs
    keep=np.isfinite(err1)&np.isfinite(err2)
    err1 = err1[keep]
    err2 = err2[keep]
    
    
    d = err1 - err2 
    if geq:
        s = (d>=0).sum()
    else:
        s = (d>0).sum()
    pval = binom.cdf(s,len(d),prob)
    if two_tailed:
        pval = binom.cdf(len(d) - s,len(d),prob) - pval
    result = SimpleNamespace(
        pval=pval,
        p_est=s/len(d),
        s=s,
        n=len(d)
    )
    return result

def diebold_mariano_correct(groundtruth,pred1,pred2,one_sided=False):
    "Actual Diebold-Mariano test"
    # kill NaNs
    
    groundtruth=np.array(groundtruth)
    pred1=np.array(pred1)
    pred2=np.array(pred2)
    
    keep=np.isfinite(pred1)&np.isfinite(pred2)&np.isfinite(groundtruth)
    pred1 = pred1[keep]
    pred2 = pred2[keep]
    groundtruth = groundtruth[keep]
    
    ordering = np.array(list(range(len(groundtruth))))
    np.random.shuffle(ordering)
    pred1=pred1[ordering]
    pred2=pred2[ordering]
    groundtruth=groundtruth[ordering]
    
    statistic,pvalue = dm_test(groundtruth,pred1,pred2, one_sided=one_sided, harvey_correction=True)
    return statistic,pvalue

def compute_pct_growth(year_start,country,df,observables):
    # find out where we have a prediction
    df = df.loc[(df.country_code==country)&(df.year_pred_start==year_start)].sort_values(by='pred_dt')
    years_with_prediction_available=[df.year_pred_start.iloc[0]] + df.year_pred_end.tolist()
    # get observable values in those years (+ the starting year)
    years_with_prediction_available = list(sorted(set(years_with_prediction_available).intersection(set(observables['gdp'].index))))
    actual_values = observables['gdp'].loc[years_with_prediction_available,country]
    # convert logarithms to unit value
    actual_values = 10**actual_values
    # normalize so that 1st year is ==1.
    pct_growth = actual_values/actual_values.iloc[0]
    # will return a timeseries with years as index
    return pct_growth

def get_forecast(year_start,country,model,df):
    # select relevant predictions
    df = df.loc[(df.year_pred_start==year_start)&(df.country_code==country)]
    # set year prediction end as index
    df = df.set_index('year_pred_end')[f'prediction_{model}']
    # add year of prediction start as 0 growth
    df.loc[year_start]=0
    # then sort by prediction end year
    df = df.sort_index()
    # add 1 to be able to exponentiate
    increase_rates = 1+df
    # extract # of years from start
    years = df.index - year_start
    # exponentiate to get predicted value
    increase_rates = increase_rates**years.values
    # will return a timeseries with years as index
    return increase_rates


def mkquiver(ax, X, Y, U, V,
             scale=1,
             arrowstyle=None, tail_width=None, head_width=None, head_length=None, linewidth=None, zorder=1,
             **kwargs
             ):
    '''
    mkquiver(ax,X,Y,U,V,
               scale = 1,
               arrowstyle='simple',
               tail_width=.005,
               head_width=.01,
               head_length=.01,
               linewidth=.5,
               edgecolor='k',
               facecolor='k'
               alpha='1',
               zorder=1,
               **kwargs
              ):

    Takes a matplotlib.axes._subplots.AxesSubplot instance as its "ax" argument, and draws
    a vector field on it. The vectors start at (X,Y) and point to (X+U,Y+V). The scale arg
    controls the scale of the  arrows, i.e. multiplies U and V. Additional kwargs can be added
    to control the appearance of the arrows. These kwargs are the same accepted by class
    matplotlib.patches.FancyArrowPatch (documented at
    http://matplotlib.org/api/patches_api.html#matplotlib.patches.FancyArrowPatch
    ).

    Useful kwargs are:

    arrowstyle: accepts a strign describing the arrow.
                '-|>' or '->' to set the arrow tip,
                fancy or simple
    tail_width: default .005
    head_width: default .01
    head_length: default .01
    alpha: default 1
    edgecolor: default 'k'
    facecolor: default 'k'
    linestyle: default ???
    zorder : set the order for drawing elements on the plot. see http://matplotlib.org/examples/pylab_examples/zorder_demo.html

    other kwargs available are:

    ===============  ======================================================
    Key              Description
    ===============  ======================================================
    arrowstyle       the arrow style
    connectionstyle  the connection style
    relpos           default is (0.5, 0.5)
    patchA           default is bounding box of the text
    patchB           default is None
    shrinkA          default is 0 points in this fcn
    shrinkB          default is 0 points in this fcn
    mutation_scale   default is text size (in points)
    mutation_aspect  default is 1.
    ?                any key for :class:`matplotlib.patches.PathPatch`
    ===============  ======================================================
    '''
    X, Y = np.array(X), np.array(Y)
    U, V = np.array(U), np.array(V)
    posx = X + (U * scale)
    posy = Y + (V * scale)
    if posx.shape != posy.shape: raise Exception(
            'X,Y,U,V are not the same size.')

    # build the `arrowstyle` arg for FancyArrowPatch, which is a string
    argument = ''
    for var, string, val in (
            (arrowstyle, '', 'simple'), (tail_width, ',tail_width=', '.005'), (head_width, ',head_width=', '.01'),
            (head_length, ',head_length=', '.01')):
        if var == None:
            argument += string + val
        elif var != None:
            argument += string + str(var)

    if linewidth != None: kwargs['linewidth'] = linewidth

    # defaults
    arrowprops = {'arrowstyle': argument,
                  'linewidth': .5,
                  'facecolor': 'k',
                  'edgecolor': 'k',
                  'shrinkA': 0,
                  'shrinkB': 0
                  }
    # override defaults
    for key in kwargs:
        arrowprops[key] = kwargs[key]

    arrows = []
    for x, y, u, v in zip(X, Y, posx, posy):
        arrows.append(matplotlib.patches.FancyArrowPatch(posA=(x, y), posB=(u, v), **arrowprops))

    arrows = matplotlib.collections.PatchCollection(arrows, match_original=True, zorder=zorder)
    ax.add_collection(arrows)

    return arrows

def plot_country_trajectories(ax, xaxis, yaxis, names, xlim=None, ylim=None, title='', xlabel='', ylabel='',
                               traj_colors=True, traj_alpha=1.0, traj_dots=False, scatter_only=False):
    """
    Plot country trajectories on an axis.
    
    Parameters:
    -----------
    ax : matplotlib axis
    xaxis : array-like, shape (n_years, n_countries)
    yaxis : array-like, shape (n_years, n_countries)
    names : list of str
        Country names for labels
    xlim, ylim : tuple or None
        Axis limits
    title, xlabel, ylabel : str
        Plot labels
    traj_colors : bool
        If True, use different colors for each country. If False, use gray.
    traj_alpha : float
        Alpha transparency for trajectories
    traj_dots : bool
        If True, also plot dots at each point (in addition to lines)
    scatter_only : bool
        If True, plot only scatter points (no lines). Overrides traj_dots.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        annotations = []
        X, Y, U, V, color = [], [], [], [], []
        for x, y, name in zip(xaxis.transpose(), yaxis.transpose(), names):
            if scatter_only:
                # Scatter only - no lines
                if traj_colors:
                    scatter = ax.scatter(x, y, s=3, alpha=traj_alpha)
                    line_color = scatter.get_facecolors()[0] if len(scatter.get_facecolors()) > 0 else 'gray'
                else:
                    ax.scatter(x, y, s=3, color='gray', alpha=traj_alpha)
                    line_color = 'gray'
            else:
                # Draw trajectory lines
                if traj_colors:
                    line = ax.plot(x, y, linewidth=.5, alpha=traj_alpha)
                    line_color = line[0].get_color()
                else:
                    line = ax.plot(x, y, linewidth=.5, color='gray', alpha=traj_alpha)
                    line_color = 'gray'
                
                if traj_dots:
                    ax.scatter(x, y, s=1, color=line_color, alpha=traj_alpha)
            
            x = x.copy()
            y = y.copy()
            if xlim is not None: x[(x<xlim[0]) | (x>xlim[1])] = np.nan
            if ylim is not None: y[(y<ylim[0]) | (y>ylim[1])] = np.nan
            startidx = np.argmax(np.isfinite(x) & np.isfinite(y))
            if startidx >= len(x)-1: startidx=0
            
            if name:  # Only annotate if name is provided
                annotation = ax.annotate(
                    xy=(x[startidx], y[startidx] + 0.03 * np.random.uniform()),
                    text=name,
                    fontsize=11,
                    color=line_color,
                    path_effects=[
                        pe.withStroke(alpha=.5, linewidth=2, foreground="white"),
                        pe.withStroke(alpha=.5, linewidth=1, foreground="black"),
                    ],
                )
                annotations.append(annotation)
            
            if np.isfinite(x[startidx] + y[startidx] + x[startidx+1] + y[startidx+1]):
                X.append(x[startidx])
                Y.append(y[startidx])
                U.append(x[startidx+1] - x[startidx])
                V.append(y[startidx+1] - y[startidx])
                color.append(line_color)
        
        if traj_colors:  # Only draw arrows if using colors
            for x, y, u, v, c in zip(X, Y, U, V, color):
                mkquiver(ax, [x], [y], [u], [v], head_width=.02, facecolor=c, linewidth=0)
        
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        return annotations

def marginal_2d(ax, X,y,x0interval,x1interval,bw=(1.,1.),resolution=100, extent=None, colorbar=False, fig=None, cmap='jet', levels=8,alpha_lines=.7,alpha_patches=.8):
    x0interval = np.linspace(*x0interval,resolution)
    x1interval = np.linspace(*x1interval,resolution)
    coords = np.meshgrid(x0interval,x1interval)
    coords = np.stack([coords[0].flatten(),coords[1].flatten()]).T
    model = NWKR(bandwidth=bw)
    model.fit(X=X,y=y)
    plotvalues, _ = model.predict(coords)
    plotvalues = plotvalues.reshape((resolution,resolution))
    contour_opts = dict()
    contour_opts['cmap']=cmap
    contour_opts['levels']=levels
    if extent is not None:
        contour_opts['vmin'], contour_opts['vmax'] = extent
    mappable = ax.contourf(x0interval,x1interval,plotvalues,alpha=alpha_patches, **contour_opts)
    contour_opts.pop('cmap')
    lines = ax.contour(x0interval,x1interval,plotvalues, colors='black', linewidths=.7, alpha=alpha_lines, **contour_opts)
    if colorbar:
        if fig is None:
            raise AssertionError('If colorbar=True, then you should pass the fig argument too!')
        cb = fig.colorbar(mappable=mappable,ax=ax)
    else:
        cb = None
    return mappable,lines,cb

def get_observables_dict_from_panel(df):
    observables = df[['year_pred_start','country_code']+[x for x in df.columns if x.startswith('value_')]].drop_duplicates()
    observables = {ob.replace('value_',''):
     observables.pivot(index='year_pred_start',columns='country_code',values=ob)
     for ob in 
     [x for x in observables.columns if x.startswith('value')]}
    for key,item in observables.items():
        observables[key] = item.loc[sorted(item.index),sorted(item.columns)]
    return observables


def NWKR_1D(X, y, new_X=None, N=100, bw=1, kernel='norm'):
    """a small wrapper to calculate 1-D NWKR for two 1-dim vectors, X and y, together with a prediction"""
    assert X.ndim == y.ndim == 1
    X = X.reshape(-1, 1)
    if new_X is None:
        new_X = np.linspace(np.nanmin(X), np.nanmax(X), num=N)
    regression = NWKR(kernel=kernel, bandwidth=bw)
    regression.fit(X, y)
    output, output_std = regression.predict(new_X.reshape(-1, 1))
    return new_X, output, output_std

def kernel_1d_plot(ax, X, y, new_X=None, N=100, bw=1, kernel='norm',
                   plot_error=True, color='black', alpha=1, alpha_error=.3,
                   plot_args={}, fill_between_args={}):
    """calculate and plot a 1-D kernel regression, with expected error, for two vectors X and y"""
    new_X, output, output_std = NWKR_1D(X, y, new_X, N, bw, kernel)
    plot = ax.plot(new_X, output, alpha=alpha, color=color, **plot_args)
    if plot_error:
        error_plot = ax.fill_between(new_X,
                                     output - output_std,
                                     output + output_std,
                                     color=color,
                                     alpha=alpha_error,
                                     **fill_between_args)
    else:
        error_plot = None
    return new_X, output, output_std, plot, error_plot

subdivisions = dict()
subdivisions['OECD']=['Australia',  'Austria',  'Belgium',  'Canada',  'Czechia',  'Denmark',  'Finland',  'France',  'Germany',  'Greece',  'Hungary',  'Iceland',  'Ireland',  'Italy',  'Japan',  'Korea, Republic of',  'Luxembourg',  'Mexico',  'Netherlands',  'New Zealand',  'Norway',  'Poland',  'Portugal',  'Slovakia',  'Spain',  'Sweden',  'Switzerland',  'Turkey',  'United Kingdom',  'United States']
subdivisions['ALL'] = ['Aruba', 'Afghanistan', 'Angola', 'Albania', 'Andorra',        'United Arab Emirates', 'Argentina', 'Armenia', 'American Samoa',        'Antigua and Barbuda', 'Australia', 'Austria', 'Azerbaijan',        'Burundi', 'Belgium', 'Benin', 'Burkina Faso', 'Bangladesh',        'Bulgaria', 'Bahrain', 'Bahamas', 'Bosnia and Herzegovina',        'Belarus', 'Belize', 'Bermuda', 'Bolivia, Plurinational State of', 'Brazil', 'Barbados', 'Brunei Darussalam', 'Bhutan', 'Botswana',        'Central African Republic', 'Canada', 'Switzerland', 'Chile',        'China', "Côte d'Ivoire", 'Cameroon',        'Congo, The Democratic Republic of the', 'Congo', 'Colombia',        'Comoros', 'Cabo Verde', 'Costa Rica', 'Cuba', 'Curaçao',        'Cayman Islands', 'Cyprus', 'Czechia', 'Germany', 'Djibouti',        'Dominica', 'Denmark', 'Dominican Republic', 'Algeria', 'Ecuador',        'Egypt', 'Eritrea', 'Spain', 'Estonia', 'Ethiopia', 'Finland',        'Fiji', 'France', 'Faroe Islands',        'Micronesia, Federated States of', 'Gabon', 'United Kingdom',        'Georgia', 'Ghana', 'Gibraltar', 'Guinea', 'Gambia',        'Equatorial Guinea', 'Greece', 'Grenada', 'Greenland', 'Guatemala',        'Guam', 'Guyana', 'Hong Kong', 'Honduras', 'Croatia', 'Haiti',        'Hungary', 'Indonesia', 'India', 'Ireland',        'Iran, Islamic Republic of', 'Iraq', 'Iceland', 'Israel', 'Italy',        'Jamaica', 'Jordan', 'Japan', 'Kazakhstan', 'Kenya', 'Kyrgyzstan',        'Cambodia', 'Kiribati', 'Saint Kitts and Nevis',        'Korea, Republic of', 'Kuwait', "Lao People's Democratic Republic",        'Lebanon', 'Liberia', 'Libya', 'Saint Lucia', 'Sri Lanka',        'Lesotho', 'Lithuania', 'Luxembourg', 'Latvia', 'Macao', 'Morocco',        'Moldova, Republic of', 'Madagascar', 'Maldives', 'Mexico',        'Marshall Islands', 'Macedonia, Republic of', 'Mali', 'Malta',        'Myanmar', 'Montenegro', 'Mongolia', 'Northern Mariana Islands',        'Mozambique', 'Mauritania', 'Mauritius', 'Malawi', 'Malaysia',        'Namibia', 'New Caledonia', 'Niger', 'Nigeria', 'Nicaragua',        'Netherlands', 'Norway', 'Nepal', 'Nauru', 'New Zealand', 'Oman',        'Pakistan', 'Panama', 'Peru', 'Philippines', 'Palau',        'Papua New Guinea', 'Poland',        "Korea, Democratic People's Republic of", 'Portugal', 'Paraguay',        'Palestine, State of', 'French Polynesia', 'Qatar', 'Romania',        'Russian Federation', 'Rwanda', 'Saudi Arabia', 'Sudan', 'Senegal',        'Singapore', 'Solomon Islands', 'Sierra Leone', 'El Salvador',        'San Marino', 'Somalia', 'Serbia', 'Sao Tome and Principe',        'Suriname', 'Slovakia', 'Slovenia', 'Sweden', 'Swaziland',        'Seychelles', 'Syrian Arab Republic', 'Turks and Caicos Islands',        'Chad', 'Togo', 'Thailand', 'Tajikistan', 'Turkmenistan', 'Tonga',        'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Tuvalu',        'Tanzania, United Republic of', 'Uganda', 'Ukraine', 'Uruguay',        'United States', 'Uzbekistan', 'Saint Vincent and the Grenadines',        'Venezuela, Bolivarian Republic of', 'Virgin Islands, British',        'Virgin Islands, U.S.', 'Viet Nam', 'Vanuatu', 'Samoa', 'Yemen', 'South Africa', 'Zambia', 'Zimbabwe']
subdivisions['ASIA'] = ['Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan', 'Brunei Darussalam', 'Cambodia', 'China', 'Cyprus', 'Georgia', 'India', 'Indonesia', 'Iran, Islamic Republic of', 'Iraq', 'Israel', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', "Lao People's Democratic Republic", 'Lebanon', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', "Korea, Democratic People's Republic of", 'Oman', 'Pakistan', 'Palestine, State of', 'Philippines', 'Qatar', 'Russian Federation', 'Saudi Arabia', 'Singapore', 'Korea, Republic of', 'Sri Lanka', 'Syrian Arab Republic', 'Taiwan', 'Tajikistan', 'Thailand', 'Timor-Leste', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen',]
subdivisions['S_AMERICA'] = ['Argentina',  'Bolivia, Plurinational State of',  'Brazil',  'Chile',  'Colombia',  'Ecuador',  'Guyana',  'Paraguay',  'Peru',  'Suriname',  'Uruguay',  'Venezuela, Bolivarian Republic of']
subdivisions['AFRICA'] = ["Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde", "Cameroon", 'Central African Republic', "Chad", "Comoros", 'Congo, The Democratic Republic of the', "Republic of the Congo", "Côte d'Ivoire", "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Swaziland", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan", "Swaziland", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe", ]
subdivisions['EU27'] = ['Austria',  'Belgium',  'Bulgaria',  'Croatia',  'Cyprus',  'Czechia',  'Denmark',  'Estonia',  'Finland',  'France',  'Germany',  'Greece',  'Hungary',  'Ireland',  'Italy',  'Latvia',  'Lithuania', 'Luxembourg',  'Malta',  'Netherlands',  'Poland',  'Portugal',  'Romania',  'Slovakia',  'Slovenia',  'Spain',  'Sweden']
subdivisions['OECD_COMPLEMENT'] = [x for x in subdivisions['ALL'] if x not in subdivisions['OECD']]
subdivisions = {key:set(item) for key,item in subdivisions.items()}


# =============================================================================
# Text Position Fixer Functions
# =============================================================================

def fix_annotations(fig, ax, annotations, correction=0.05):
    """
    Fix annotations that are outside the plot boundaries.
    """
    # Get the plot's boundaries
    left_boundary, right_boundary = ax.get_xlim()
    bottom_boundary, top_boundary = ax.get_ylim()
    
    # Get the renderer
    renderer = fig.canvas.get_renderer()
    
    # Annotate with adjustment for out-of-bounds text
    for ann in annotations:
        # Get the bounding box of the annotation
        bbox = ann.get_window_extent(renderer=renderer)
        bbox_data = bbox.transformed(ax.transData.inverted())
        
        x, y = ann.xy
        dx, dy = 0, 0
        
        # Check and adjust for right boundary
        if bbox_data.x1 > right_boundary:
            dx = right_boundary - bbox_data.x1 - correction
        
        # Check and adjust for left boundary
        elif bbox_data.x0 < left_boundary:
            dx = left_boundary - bbox_data.x0 + correction
        
        # Check and adjust for top boundary
        if bbox_data.y1 > top_boundary:
            dy = top_boundary - bbox_data.y1 - correction
        
        # Check and adjust for bottom boundary
        elif bbox_data.y0 < bottom_boundary:
            dy = bottom_boundary - bbox_data.y0 + correction
        
        # Update the annotation position if adjustment is needed
        if dx != 0 or dy != 0:
            new_x = x + dx
            new_y = y + dy
            ann.set_position((new_x, new_y))


def adjust_text_positions(texts, ax, expand=1.1, iterations=100):
    """
    Adjust text positions to reduce overlap within the axes coordinate system.
    
    :param texts: List of matplotlib.text.Text objects
    :param ax: The matplotlib Axes object
    :param expand: Factor to expand the bounding box of texts
    :param iterations: Number of iterations to attempt adjustment
    """
    def get_text_position(text):
        return np.array(text.get_position())
    
    def set_text_position(text, pos):
        text.set_position(pos)
    
    def get_text_bounding_box(text):
        bbox = text.get_window_extent(renderer=ax.figure.canvas.get_renderer())
        return ax.transData.inverted().transform(bbox)
    
    def check_overlap(bbox1, bbox2):
        return (bbox1[0, 0] < bbox2[1, 0] and bbox1[1, 0] > bbox2[0, 0] and
                bbox1[0, 1] < bbox2[1, 1] and bbox1[1, 1] > bbox2[0, 1])

    positions = np.array([get_text_position(text) for text in texts])
    
    for _ in range(iterations):
        moved = False
        for i, text1 in enumerate(texts):
            bbox1 = get_text_bounding_box(text1)
            center = (bbox1[0] + bbox1[1]) / 2
            size = bbox1[1] - bbox1[0]
            bbox1 = np.array([center - size * expand / 2, center + size * expand / 2])
            
            for j, text2 in enumerate(texts[i+1:], start=i+1):
                bbox2 = get_text_bounding_box(text2)
                if check_overlap(bbox1, bbox2):
                    direction = positions[j] - positions[i]
                    distance = np.linalg.norm(direction)
                    if distance > 0:
                        direction /= distance
                        move = min(distance * 0.1, 0.01)  # Limit the movement
                        positions[i] -= direction * move
                        positions[j] += direction * move
                        moved = True
        
        if not moved:
            break

    for text, pos in zip(texts, positions):
        set_text_position(text, pos)


def general_text_position_fixer(fig, ax, texts, cycles=20, expand=1.1, iterations=5, correction=0.05):
    """
    Fix text positions by iteratively adjusting for overlap and boundary violations.
    """
    for _ in range(cycles):
        adjust_text_positions(texts=texts, ax=ax, iterations=iterations, expand=expand)
        fix_annotations(fig=fig, ax=ax, annotations=texts, correction=correction)
