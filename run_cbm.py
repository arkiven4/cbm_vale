import os, pickle, sqlite3, copy, time, sklearn, sys, clr
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy.signal import resample

from src.models import *
from src.utils import *
from main import  load_dataset, backprop
import src.commons as commons
import src.custom_const as custom_const
from src.commons import OnlinePercentileEstimator

sys.path.append(r'C:\Program Files (x86)\PIPC\AF\PublicAssemblies\4.0')  
clr.AddReference('OSIsoft.AFSDK')

from OSIsoft.AF import *  
from OSIsoft.AF.PI import *
from OSIsoft.AF.Search import * 
from OSIsoft.AF.Asset import *  
from OSIsoft.AF.Data import *  
from OSIsoft.AF.Time import *  
from OSIsoft.AF.UnitsOfMeasure import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
torch.zero_grad = True

import sys
sys.path.append('./accumulation_tree/accumulation_tree')
from accumulation_tree import AccumulationTree

def getdf_piserver(piServer, pi_tag, time_list):
    timerange = AFTimeRange(time_list[0], time_list[1])
    master_pd = ""
    for i in range(len(pi_tag)):
        tag = PIPoint.FindPIPoint(piServer, pi_tag[i])
        value_resp = parse_recorded_events(tag.InterpolatedValues(timerange, AFTimeSpan.Parse('1m'), '', False))
        if i == 0:
            value_resp['Timestamps'] = pd.to_datetime(value_resp['Timestamps'])
            master_pd = value_resp
        else:
            master_pd = pd.concat([master_pd, value_resp['Values']], axis=1, join='inner')

    master_pd = master_pd.values
    master_pd = pd.DataFrame(data=master_pd, columns=['TimeStamp'] + feature_set + ['Grid Selection'])
    master_pd.replace('I/O Timeout', np.nan, inplace=True)
    master_pd.replace('No Data', np.nan, inplace=True)
    master_pd.replace('Future Data Unsupported', np.nan, inplace=True)
    master_pd.replace('Closed', np.nan, inplace=True)
    master_pd.replace('Open', np.nan, inplace=True)
    for column_name in master_pd.columns:
        if column_name != 'Load_Type' and column_name != 'TimeStamp':
            master_pd[column_name] = pd.to_numeric(master_pd[column_name], downcast='float')
    master_pd = master_pd.sort_values(by='TimeStamp')
    master_pd = master_pd.reset_index(drop=True)
    master_pd = master_pd.fillna(method='ffill')

    df_sel = master_pd.iloc[-120:, :]
    df_sel = df_sel.reset_index(drop=True)
    
    df_additional = df_sel[['Grid Selection']].copy()
    df_additional = df_additional.astype(float)
    df_sel = df_sel[['TimeStamp'] + feature_set] 
    return df_sel, df_additional

############################ Configuration ###############################
measured_horizon = 60 * 2 * 1 # Minute
interval_gap = 10 * 60 # Seconds

# Update Thr Each
total_days = 30
total_minutes = total_days * 24 * 60
count_accumulateArrayForCalc = total_minutes * 60 // interval_gap

feature_set = ['Active Power', 'Reactive Power', 'Governor speed actual', 'UGB X displacement', 
               'UGB Y displacement', 'LGB X displacement', 'LGB Y displacement', 'TGB X displacement',
               'TGB Y displacement', 'Stator winding temperature 13',
               'Stator winding temperature 14', 'Stator winding temperature 15',
               'Surface Air Cooler Air Outlet Temperature',
               'Surface Air Cooler Water Inlet Temperature',
               'Surface Air Cooler Water Outlet Temperature',
               'Stator core temperature', 'UGB metal temperature',
               'LGB metal temperature 1', 'LGB metal temperature 2',
               'LGB oil temperature', 'Penstock Flow', 'Turbine flow',
               'UGB cooling water flow', 'LGB cooling water flow',
               'Generator cooling water flow', 'Governor Penstock Pressure',
               'Penstock pressure', 'Opening Wicked Gate', 'UGB Oil Contaminant',
               'Gen Thrust Bearing Oil Contaminant']
tag_array = [custom_const.feature_tag_mapping[feature] for feature in feature_set + ['Grid Selection']]

# Model used
model_array = ["Attention", "DTAAD", "MAD_GAN", "TranAD", "DAGMM", "USAD", "OmniAnomaly"]
############################ Setup ###############################
commons.init_db_timeconst(feature_set, "db/original_data.db", "original_data")
commons.init_db_timeconst(['Grid Selection'], "db/original_data.db", "additional_original_data")
commons.init_db_timeconst(feature_set, "db/severity_trendings.db", "severity_trendings")
commons.init_db_timeconst(feature_set, "db/severity_trendings.db", "original_sensor")
for model_name in model_array:
    commons.init_db_timeconst(feature_set, "db/pred_data.db", model_name)
    commons.init_db_timeconst(feature_set, "db/threshold_data.db", model_name)
    commons.init_db_timeconst(feature_set, "db/adaptive_tdigest.db", model_name)

############################ Connect PI Server ###############################
piServers = PIServers()
piServer = piServers["PTI-PI"]
piServer.Connect(False) 
print ('Connected to server: ' + "PTI-PI")

############################ Load Models And Variable ###############################
with open('normalize_2023.pickle', 'rb') as handle:
    normalize_obj = pickle.load(handle)
    min_a, max_a = normalize_obj['min_a'], normalize_obj['max_a']

tdigest_models = {}
for model_now in model_array:
    path_try = f'mini_loss_fold/{args.dataset}/{model_now}_tdigest_run.pickle'
    path_fallback = f'mini_loss_fold/{args.dataset}/{model_now}_tdigest.pickle'
    with open(path_try if os.path.exists(path_try) else path_fallback, 'rb') as handle:
        tdigest_models[model_now] = pickle.load(handle)

loss_accumulative_path = "mini_loss_fold/loss_accumulative.pickle"
if os.path.exists(loss_accumulative_path):
    with open(loss_accumulative_path, 'rb') as handle:
        loss_accumulative = pickle.load(handle)
else:
    loss_accumulative = {model_now: [[] for _ in feature_set] for model_now in model_array}

############################ Main Loops  ###############################
count = 0
while True:
    valid_measurment = True
    threshold_percentages = {}
    ypred_models = {}

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Executing CBM Task #{count}...")
    now_time = datetime.utcnow()
    start_time = now_time - timedelta(minutes=measured_horizon)
    time_list = [start_time.strftime('%Y-%m-%d %H:%M:%S'), now_time.strftime('%Y-%m-%d %H:%M:%S')]
    df_sel, df_additional = getdf_piserver(piServer, tag_array, time_list)
    if len(df_sel) <= 0:
        count = count + 1
        time.sleep(interval_gap)
        continue
    
    load_label = df_sel.apply(commons.label_load, axis=1).value_counts()
    bad_pct = (load_label.get('No Load', 0) +  load_label.get('Shutdown', 0)) / load_label.sum()
    testD, testO, df_timestamp, df_feature = commons.preprocessPD_loadData(df_sel, feature_set, min_a, max_a)
    for feat_index, feature_now in enumerate(feature_set):
        if (df_feature[feature_set] == 0).any().any():
            valid_measurement = False

    temploss_models = {model_now: None for model_now in model_array}
    for model_now in model_array:
        model = commons.load_model(args.dataset, model_now, testO.shape[1], args.retrain, args.test)
        model.eval()
        if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN', 'TranAD'] or 'DTAAD' in model.name:
            testD_now = commons.convert_to_windows(testD, model)
        else:
            testD_now = testD
        loss, y_pred = backprop(0, model, testD_now, testO, None, None, training=False)
        y_pred = np.where(np.isfinite(y_pred), y_pred, np.finfo(np.float32).eps)
        temploss_models[model_now] = loss
        ypred_models[model_now] = commons.denormalize3(y_pred, min_a, max_a)
        threshold_percentages[model_now] = commons.calcThres_oneModel(feature_set, loss, tdigest_models[model_now])

    counter_feature_trd, _ = commons.calc_counterPercentage(threshold_percentages, feature_set, model_array)
    if valid_measurement and bad_pct == 0.0:
        for feat_index, feature_now in enumerate(feature_set):
            if counter_feature_trd[feature_now]['percentage'] <= 1.0 and counter_feature_trd[feature_now]['count'] == 0:
                for model_now in model_array:
                    loss_accumulative[model_now][feat_index].append(temploss_models[model_now][:, feat_index])

    df_feature = commons.denormalize3(df_feature, min_a, max_a)
    df_feature_mean = commons.trunc(df_feature.values.mean(axis=0), decs=2)
    df_feature = df_feature.values[::6]
    df_additional = df_additional.values[::6]
    df_timestamp = df_timestamp.dt.floor("min")[::6].values[:20]
    for model_now in model_array:
        ypred_models[model_now] = ypred_models[model_now][::6]

    min_len = min(len(df_timestamp), len(df_feature), *map(len, ypred_models.values()))
    df_timestamp = df_timestamp[:min_len]
    df_feature = df_feature[:min_len]
    df_additional = df_additional[:min_len]
    for model_now in model_array:
        ypred_models[model_now] = ypred_models[model_now][:min_len]

    mask = df_timestamp > df_timestamp_last
    df_feature = df_feature[mask]
    df_additional = df_additional[mask]
    df_timestamp = df_timestamp[mask]
    for model_now in model_array:
        ypred_models[model_now] = ypred_models[model_now][mask]

    if len(df_timestamp) <= 0:
        count = count + 1
        time.sleep(interval_gap)
        continue

    df_timestampi = pd.to_datetime(df_timestamp[-1])
    trend_data = np.array([counter_feature_trd[key]['percentage'] for key in counter_feature_trd]).astype(np.float64)

    commons.batch_timeseries_savedb(df_timestamp, commons.trunc(df_feature, decs=2), feature_set, "db/original_data.db", "original_data")
    commons.batch_timeseries_savedb(df_timestamp, trunc(df_additional, decs=2), ['Grid Selection'], "db/original_data.db", "additional_original_data")
    commons.timeseries_savedb(df_timestampi, trend_data, feature_set, "db/severity_trendings.db", "severity_trendings")
    commons.timeseries_savedb(df_timestampi, df_feature_mean, feature_set, "db/severity_trendings.db", "original_sensor")
    for idx_model, (model_name) in enumerate(model_array):
        commons.batch_timeseries_savedb(df_timestamp, commons.trunc(ypred_models[model_name], decs=2), feature_set, "db/pred_data.db", model_name)
        commons.timeseries_savedb(df_timestampi, commons.trunc(np.array(list(threshold_percentages[model_name].values())), decs=2), feature_set, "db/threshold_data.db", model_name)
        commons.timeseries_savedb(df_timestampi, commons.trunc(np.array([tdigest_models[model_name][index].get_percentile(99) for index in range(len(feature_set))]), decs=6), feature_set, "db/adaptive_tdigest.db", model_name)

    for model_now in model_array:
        for feat_index in range(len(feature_set)):
            if len(loss_accumulative[model_now][feat_index]) >= count_accumulateArrayForCalc:
                value_toupdate = np.concatenate(loss_accumulative[model_now][feat_index], axis=0)
                tdigest_models[model_now][feat_index].update(value_toupdate)
                loss_accumulative[model_now][feat_index] = []
        with open(f'mini_loss_fold/{args.dataset}/{model_now}_tdigest_run.pickle', 'wb') as handle:
            pickle.dump(tdigest_models[model_now], handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'mini_loss_fold/loss_accumulative.pickle', 'wb') as handle:
        pickle.dump(loss_accumulative, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # DONT REMOVE THIS
    df_timestamp_last = df_timestamp[-1]
    count = count + 1
    time.sleep(interval_gap)


