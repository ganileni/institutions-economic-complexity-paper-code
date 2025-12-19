# Here all calculations are done in log-diffs, then final predictions are converted to CAGR.
from pathlib import Path
import functools
import pandas as pd
import numpy as np
from ectools.SPSb import *
from ectools.utilities import log2cagr_transf, make_cagr_diff
from collections import defaultdict
import itertools


# data cleaning functions
def keep_common_year_countries(observables):
    common_years = sorted(functools.reduce(lambda x,y: x&y, [set(item.index) for key,item in observables.items()]))
    common_countries = sorted(functools.reduce(lambda x,y: x&y, [set(item.columns) for key,item in observables.items()]))
    for key,item in observables.items():
        observables[key] = item.loc[common_years,common_countries]

# take logarithms
def take_logarithms(observables,skip={}):
    for key,item in observables.items():
        if key in skip: continue
        observables[key] = item.apply(np.log10)
    delete_inf_data(observables)

def delete_inf_data(observables):
    """Apply after taking logarithms because they can result in inf's"""
    for key,item in observables.items():
        item.replace([np.inf, -np.inf], np.nan, inplace=True)

def tech_fitness_last_year_fix(df):
    """Last year of tech fitness (2017) is all NaNs except for 2 countries. If you leave it at NaN, the whole of 2017 will be removed from the calculations basically. If you copy the 2016 values onto 2017, they will not be used to make any backtesting, but the predictions won't be thrown out. (it will influence forecasts though)."""
    df = df.copy()
    copy_values = np.isnan(df.values[-1,:])
    df.iloc[-1,:][copy_values] = df.values[-2,:][copy_values]
    return df

def delete_missing_data(observables):
    keep_data = np.ones(observables[list(observables.keys())[0]].values.shape,dtype=bool)
    # generate a mask
    for key,item in observables.items():
        keep_data = keep_data & np.isfinite(item.values).astype(bool)
    # invert_mask
    kill_mask = ~keep_data
    for key,item in observables.items():
        values = item.values
        values[kill_mask] = np.nan
        item.iloc[:,:] = values


def delete_below_fitness_threshold(observables,threshold=-6.):
    try:
        observables['fitness'][observables['fitness']<threshold]=np.nan
    except KeyError:
        pass
    
def interpolate_missing_values(observables):
    for key,item in observables.items():
        observables[key] = item.interpolate(method='linear')
    

# prediction computation functions
def make_predictions_dt(predictors,to_predict,bw,dt,observables):
    """Just a giant sequence of loops to transform the data into panel format (from vaguely longitudinal).
    Not very elegant, but it does the job.
    """
    iterator = dt_predictor_iterator(
                         trajectories=[observables[x].values for x in predictors],
                         y=observables[to_predict].values,
                         dt=dt,
                         diff_fcn = lambda t0,t1: t1-t0,
                         start_period=observables[to_predict].index[0],
                         min_train_len=dt,
    )
    iterator = list(iterator)
    
    def rescaler(j):
        return (10**(j/dt))-1
    
    result=list()
    autocor_pred = list()
    velocity_spsb_list = list()
    for X,Y in iterator:
        result.append(backtest_SPSb(X,Y,bw=bw,compute_train=False))
        autocor_pred.append(autocorrelation_prediction(Y,ignore_missing_data=True))
        velocity_spsb_list.append(velocity_spsb(result[-1],Y,vel_dt_sigma=1))
        
    
    year_pred_start = list()
    year_pred_end = list()
    pred = list()
    groundtruth = list()
    std = list()
    last_observed_velocity = list()
    last_observed_velocity_gt = list()
    velocity_spsb_prediction = list()
    velocity_spsb_std = list()
    for (X,Y),res,a_p,vspsb in zip(iterator,result,autocor_pred,velocity_spsb_list):
        assert len(Y.d_test_periods_tuples)==1
        pred_tuple = Y.d_test_periods_tuples[0]
        year_pred_start.append(pred_tuple[0])
        year_pred_end.append(pred_tuple[1])
        # we are predicting one year at a time,
        # using all the available data before it
        # so shape[0] (time dimension) must be 1
        assert res.pred.shape[0]==1
        pred.append(rescaler(res.pred.flatten()))
        groundtruth.append(rescaler(res.groundtruth.flatten()))
        std.append(rescaler(res.std.flatten()))
        last_observed_velocity.append(rescaler(a_p.pred))
        last_observed_velocity_gt.append(rescaler(a_p.groundtruth))
        if not vspsb.pred.shape[0]==1:
            # sometimes you don't have enough data to compute last observed velocity
            # in this case, put in a filler of NaNs
            vspsb.pred = np.empty((1,vspsb.pred.shape[1]))
            vspsb.pred[:] = np.nan
            vspsb.std = vspsb.pred
        assert vspsb.pred.shape[0]==1
        velocity_spsb_prediction.append(rescaler(vspsb.pred.flatten()))
        velocity_spsb_std.append(rescaler(vspsb.std.flatten()))
            

    pred = np.vstack(pred)
    groundtruth = np.vstack(groundtruth)
    std = np.vstack(std)
    velocity_spsb_prediction = np.vstack(velocity_spsb_prediction)
    velocity_spsb_std = np.vstack(velocity_spsb_std)

    predictors_string = '-'.join(sorted(predictors))

    records = list()
    for yps,ype,p_y,g_y,s_y,lov,lov_gt,vspsb_p_y,vspsb_s_y in zip(
        year_pred_start,year_pred_end,pred,groundtruth,std,last_observed_velocity,last_observed_velocity_gt,velocity_spsb_prediction,velocity_spsb_std):
        if lov.shape[0]:
            lov_iterator = iter(lov.flatten())
        if lov_gt.shape[0]:
            lov_gt_iterator = iter(lov_gt.flatten())
            
        for country_code,p,g,s,vp,vs in zip(observables[to_predict].columns,p_y,g_y,s_y,vspsb_p_y,vspsb_s_y):
            records.append({
                'year_pred_start':yps,
                'year_pred_end':ype,
                'country_code':country_code,
                f'prediction':p,
                f'groundtruth':g,
                f'std':s,
                f'last_observed_velocity':next(lov_iterator) if lov.shape[0] else np.nan,
                f'last_observed_velocity_gt':next(lov_gt_iterator) if lov_gt.shape[0] else np.nan,
                f'prediction_vspsb':vp,
                f'std_vspsb':vs,
            }
            )

    records = pd.DataFrame.from_records(records)
    # drop non available predictions
    keep = np.isfinite(records[f'prediction'])
    records = records.loc[keep,:]
    return records

def make_predictions_at_all_dts(predictors,to_predict,bw,observables):
    """Just looping make_predictions_dt() over all possible dts and concatenating"""
    max_dt = int(np.floor((observables[to_predict].index.max()-observables[to_predict].index.min())/2))
    if max_dt == int((observables[to_predict].index.max()-observables[to_predict].index.min())/2):
        max_dt -= 1
    results = list()
    for dt in range(1,max_dt+1):
        _df = make_predictions_dt(predictors,to_predict,bw,dt,observables)
        _df['pred_dt'] = [dt]*_df.shape[0]
        results.append(_df)
    results = pd.concat(results)
    return results

def pred_hash(yps,ype,country_code,dt):
    """Each prediction can be assigned a unique identifier by the year it starts, the year it ends and which country we are predicting.
    Will be used as unique key to join dataframes with possibly different length later on.
    note dt is redundant information, left for readability."""
    return f'{yps}{ype}{country_code}{dt}'

def add_prediction_hashes(df):
    hashes = list()
    for yps,ype,cc,dt in zip(df.year_pred_start,df.year_pred_end,df.country_code,df.pred_dt):
        hashes.append(pred_hash(yps,ype,cc,dt))
    df.index = hashes
    return df

def make_predictions_at_all_dts_on_predictors(all_models,bws,to_predict,observables):
    "Loops make_predictions_dt() over all predictors combinations in `predictors`. Kernel bandwidths are assigned with the dictionary `bws`. Quantity to predict is assigned with `to_predict`."
    all_results = dict()
    for predictors in all_models:
        predictors = list(sorted(predictors))
        model_string = '-'.join(predictors)
        bw = [bws[p] for p in predictors]
        all_results[model_string] = make_predictions_at_all_dts(predictors,to_predict,bw,observables)
        all_results[model_string] = add_prediction_hashes(all_results[model_string])

    # real bottleneck of calculation: aggregating data. doing it by prediction hash.
    # There's a faster way making a single dataframe with the repeated columns,
    # then individual dataframes with non-repeated columns, and reducing with
    # pd.DataFrame.join()
    repeated_columns = ['year_pred_start', 'year_pred_end', 'country_code',
           'groundtruth', 'last_observed_velocity', 'pred_dt']
    ignored_columns = ['last_observed_velocity_gt']
    nonrepeated_columns = ['prediction', 'std', 'prediction_vspsb', 'std_vspsb']
    mixed_results = defaultdict(dict)
    for key,item in all_results.items():
        df = item.copy()
        for column in nonrepeated_columns:
            df[f'{column}_{key}'] = df[f'{column}']
            df = df.drop(columns=column)
        df = df.drop(columns=ignored_columns)
        df = {i:r for i,r in zip(df.index,df.to_dict('records'))}
        for key,item in df.items():
            mixed_results[key].update(item)
    mixed_results = pd.DataFrame(mixed_results).T
    return mixed_results

def observable_to_dict(observable_df):
    dictionary = dict()
    for y, countries in observable_df.iterrows():
        dictionary[y] = {country:value for country,value in zip(countries.index,countries.values)}
    return dictionary

def add_observable_values_to_df(df,observables):
    for obs, obs_df in observables.items():
        obs_dict = observable_to_dict(obs_df)
        obs_panel = [obs_dict[year][country] for year,country in zip(df.year_pred_start,df.country_code)]
        df[f'value_{obs}'] = obs_panel
    return df


def format_dataframe(predictions,observables,quantity_to_predict,prediction_start_year):
    predictions = pd.DataFrame.from_dict(predictions).T
    predictions = predictions.copy()
    predictions.columns = observables[quantity_to_predict].columns
    predictions = predictions.melt(var_name='country_code',ignore_index=False)
    predictions['year_pred_end'] = predictions.index
    predictions['year_pred_start'] = prediction_start_year
    predictions['pred_dt'] = predictions['year_pred_end']-predictions['year_pred_start']
    predictions = add_prediction_hashes(predictions)
    predictions = predictions.sort_index()
    return predictions

def make_predictions(obs_order,bw_dict,y_vel,quantity_to_predict,max_dt,prediction_start_year,observables):
    def format_dataframe_partial(predictions):
        return format_dataframe(predictions,observables,quantity_to_predict,prediction_start_year)
    models = dict()
    predictions = dict()
    stds = dict()
    last_observed_velocity = dict()
    velocity_sigma = dict()
    for dt in range(1,max_dt+1):
        y = y_vel[dt]
        models[dt] = SPSb(
            trajectories = [observables[o].values[:y.shape[0],:] for o in obs_order],
            y_vel = y,
            bandwidth = [bw_dict[o] for o in obs_order]
            )
        predictions[dt+prediction_start_year], stds[dt+prediction_start_year] = \
            models[dt].predict([observables[o].values[-1,:] for o in obs_order])
        last_observed_velocity[dt+prediction_start_year]=y_vel[dt][-1,:]
        velocity_sigma[dt+prediction_start_year] = np.nanstd(y_vel[1],axis=0)
    predictions = format_dataframe_partial(predictions)
    stds = format_dataframe_partial(stds)
    last_observed_velocity = format_dataframe_partial(last_observed_velocity)
    velocity_sigma = format_dataframe_partial(velocity_sigma)
    return predictions,stds,last_observed_velocity,velocity_sigma

