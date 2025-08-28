import torch, os
import pickle, os, sqlite3, copy, sklearn
from datetime import datetime
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
 
def label_load(row):
   if row['Active Power'] < 1 and row['Governor speed actual'] < 1:
      return 'Shutdown'
   elif row['Active Power'] < 3 and row['Governor speed actual'] < 250:
      return 'Warming'
   elif row['Active Power'] < 3 and row['Governor speed actual'] > 250:
      return 'No Load'
   elif row['Active Power'] >= 1 and row['Active Power'] < 20 and row['Governor speed actual'] > 250:
      return 'Low Load'
   elif row['Active Power'] >= 20 and row['Active Power'] < 40 and row['Governor speed actual'] > 250:
      return 'Rough Zone'
   elif row['Active Power'] >= 40 and row['Active Power'] < 50 and row['Governor speed actual'] > 250:
      return 'Part Load'
   elif row['Active Power'] >= 50 and row['Active Power'] < 65 and row['Governor speed actual'] > 250:
      return 'Efficient Load'
   elif row['Active Power'] >= 65 and row['Governor speed actual'] > 250:
      return 'High Load'
   else:
      return 'Undefined'
   
def normalize3(a, min_a=None, max_a=None):
    if min_a is None: min_a, max_a = np.min(a, axis=0), np.max(a, axis=0)
    return ((a - min_a) / (max_a - min_a + 0.0001)), min_a, max_a

def denormalize3(a_norm, min_a, max_a):
    return a_norm * (max_a - min_a + 0.0001) + min_a

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def init_db_timeconst(feature_set, db_name="masters_data.db", table_name="severity_trending"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Create table if it does not exist
    columns = ", ".join([feature_name.replace(" ", "_") for feature_name in feature_set])
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            {columns})
    """)

    conn.commit()
    conn.close()

def batch_timeseries_savedb(df_timestamps, data, feature_set, db_name="data.db", table_name="sensor_data"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Convert timestamps to ISO format
    timestamps = [pd.to_datetime(ts).isoformat() for ts in df_timestamps]
    
    # Build column names for features, replacing spaces with underscores
    feature_columns = ', '.join([feature_name.replace(" ", "_") for feature_name in feature_set])
    placeholders = ', '.join(['?' for _ in range(len(feature_set)+1)])  # 30 features + 1 timestamp
    
    # Prepare batch data
    batch_data = [(timestamps[i], *data[i]) for i in range(data.shape[0])]
    
    # Upsert using INSERT OR REPLACE (Ensure UNIQUE constraint on timestamp in your DB schema)
    sql = f"""
        INSERT OR REPLACE INTO {table_name} (timestamp, {feature_columns})
        VALUES ({placeholders})
    """
    
    cursor.executemany(sql, batch_data)
    conn.commit()
    conn.close()

def timeseries_savedb(df_timestamp, data, feature_set, db_name="data.db", table_name="sensor_data"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # Generate timestamp
    timestamp = df_timestamp.isoformat()
    
    # Build column names for features, replacing spaces with underscores
    feature_columns = ', '.join([feature_name.replace(" ", "_") for feature_name in feature_set])
    placeholders = ', '.join(['?' for _ in range(len(feature_set))])
    
    # Upsert using INSERT OR REPLACE
    # Note: Your table must have a UNIQUE constraint on the timestamp column.
    sql = f"""
        INSERT OR REPLACE INTO {table_name} (timestamp, {feature_columns})
        VALUES (?, {placeholders})
    """
    cursor.execute(sql, (timestamp, *data))
    
    conn.commit()
    conn.close()

def fetch_between_dates(start_date, end_date, db_name="data.db", table_name="sensor_data"):
    start_date = start_date.replace(" ", "T")
    end_date = end_date.replace(" ", "T")
    
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    cursor.execute(f"""
        SELECT * FROM {table_name} WHERE timestamp BETWEEN ? AND ?
    """, (start_date, end_date))
    
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return np.array([])
    
    return np.array(rows)

def convert_timestamp(timestamp_str):
    dt = datetime.fromisoformat(timestamp_str)
    return pd.Timestamp(dt.strftime('%Y-%m-%d %H:%M:%S'))

def convert_to_windows(data, model):
    windows = []
    w_size = model.n_window
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i - w_size:i]  # cut
        else:
            w = torch.cat([data[0].repeat(w_size - i, 1), data[0:i]])  # pad
        windows.append(w if 'DTAAD' in model.name or 'Attention' in model.name or 'TranAD' in model.name else w.view(-1))
    return torch.stack(windows)

def load_model(dataset, modelname, dims, retrainMode=False, testMode=False):
    import src.models as models
    model_class = getattr(models, modelname)
    model = model_class(dims).double()
    fname = f'checkpoints/{model.name}_{dataset}/model.ckpt'
    if os.path.exists(fname) and (not retrainMode or testMode):
        checkpoint = torch.load(fname, weights_only=False, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Creating new model: {model.name}")
        assert True
    return model

def filter_noise_es(df, alpha=0.4, reduction=False):
    import copy
    new_df = copy.deepcopy(df)
    
    for column in df:
        new_df[column] = df[column].ewm(alpha=alpha, adjust=False).mean()
    
    if reduction:
        return new_df[::len(df)]  # Adjust sparsity if needed
    else:
        return new_df

def wgn_pandas(df_withtime, snr, alpha=0.15, window_size=120):
    df_no_timestamp = df_withtime.drop(columns=['TimeStamp'])
    noisy_df = pd.DataFrame(index=df_no_timestamp.index, columns=df_no_timestamp.columns)

    for start in range(0, len(df_no_timestamp), window_size):
        window = df_no_timestamp.iloc[start:start + window_size]
        
        Ps = np.sum(np.power(window, 2), axis=0) / len(window)
        Pn = Ps / (np.power(10, snr / 10))

        noise = np.random.randn(*window.shape) * np.sqrt(Pn.values)
        noisy_window = window + (noise / 100)

        noisy_df.iloc[start:start + window_size] = noisy_window
    
    noisy_df.reset_index(drop=True, inplace=True)
    noisy_df = filter_noise_es(pd.DataFrame(noisy_df, columns=noisy_df.columns), alpha)

    df_timestamp = df_withtime['TimeStamp']
    df_timestamp.reset_index(drop=True, inplace=True)

    df_withtime = pd.concat([df_timestamp, noisy_df], axis=1)
    return df_withtime

def preprocessPD_loadData(df_sel, feature_set, min_a, max_a):
    df_sel = wgn_pandas(df_sel, 30, alpha=0.15)

    df_timestamp = df_sel.iloc[:, 0]
    df_feature =  df_sel.iloc[:, 1:]
    df_feature = df_feature[feature_set]

    df_feature, _, _ = normalize3(df_feature, min_a, max_a)
    df_feature = df_feature.astype(float)

    test_loader = DataLoader(df_feature.values, batch_size=df_feature.shape[0])
    testD = next(iter(test_loader))
    testO = testD
    return testD, testO, df_timestamp, df_feature

def update_statisticGlobal(mean_global, M2_global, n_total, mean_i, std_i, n_i):
    delta = mean_i - mean_global
    new_total = n_total + n_i
    mean_global += delta * n_i / new_total
    M2_global += std_i**2 * (n_i - 1)
    M2_global += delta**2 * n_total * n_i / new_total
    return mean_global, M2_global, new_total

def calcThres_oneModel(feature_set, current_loss, now_tdigest_model):
    epsilon = np.finfo(np.float64).eps
    temp_severity = {}
    for sensor_idx in range(current_loss.shape[-1]):
        now_loss = current_loss[:, sensor_idx]

        loss_thre = now_tdigest_model[sensor_idx].get_percentile(99)
        loss_threVector = np.full(len(now_loss), loss_thre)
        loss_aboveThre = np.maximum(now_loss, loss_threVector)

        sever_comp1 = np.sum((now_loss > loss_thre)) / len(now_loss)
        sever_comp2 = np.mean(np.abs((loss_threVector - loss_aboveThre) / (loss_threVector + epsilon)))
        if sever_comp2 > 1:
            sever_comp2 = 1

        severity_percentage = (sever_comp1 * sever_comp2) * 100
        temp_severity[feature_set[sensor_idx]] = round(float(severity_percentage), 2)

    return temp_severity

def calc_counterPercentage(threshold_percentages, feature_set, model_array, calc_featureplot=False):
    mean_severity_percentage = {f: {"count": 0, "percentage": 0} for f in feature_set}

    # Accumulate count and percentage
    for values_pred in threshold_percentages.values():
        for feature_name, p in values_pred.items():
            if p >= 20:
                mean_severity_percentage[feature_name]["count"] += 1
            mean_severity_percentage[feature_name]["percentage"] += p

    # Normalize or reset based on count
    for f, v in mean_severity_percentage.items():
        if v["count"] >= 2:
            v["count"] = round((v["count"] / len(model_array)) * 100, 2)
            v["percentage"] = round(v["percentage"] / len(model_array))
        else:
            v["count"] = 0
            v["percentage"] = 0

    #mean_severity_percentage = dict(sorted(mean_severity_percentage.items(), key=lambda item: item[1]['percentage'], reverse=True))
    counter_feature_plot = {}
    if calc_featureplot:
        # Find Which Model Have Highest Confidence
        for index, value in mean_severity_percentage.items():
            higher_data = {"model": 0, "percentage": 0}
            for model_idx in threshold_percentages:
                if index in threshold_percentages[model_idx]:
                    if higher_data["percentage"] <= threshold_percentages[model_idx][index]:
                        higher_data["model"] = model_idx
                        higher_data["percentage"] = threshold_percentages[model_idx][index]
            
            counter_feature_plot[index] = higher_data['model']

    return mean_severity_percentage, counter_feature_plot

def process_shutdown_and_snl_periods(df_selected, column_name):
    data_timestamp = df_selected[['TimeStamp']].values
    sensor_datas = df_selected[column_name].values

    activepower_data = sensor_datas[:, 0].astype(float)
    rpm_data = sensor_datas[:, 1].astype(float)

    shutdown_mask = (activepower_data <= 3) & (rpm_data <= 10)
    snl_mask = (activepower_data <= 3) & (rpm_data >= 259.35) & (rpm_data <= 286.65)

    def extract_periods(mask):
        change_points = np.diff(mask.astype(int), prepend=0)
        start_indices = np.where(change_points == 1)[0]
        end_indices = np.where(change_points == -1)[0]

        if mask[-1]:
            end_indices = np.append(end_indices, len(mask))
        if mask[0]:
            start_indices = np.insert(start_indices, 0, 0)

        periods = []
        for start, end in zip(start_indices, end_indices):
            start_time = data_timestamp[start][0]
            end_time = data_timestamp[end - 1][0]
            periods.append((start_time, end_time))
        return periods

    shutdown_periods = extract_periods(shutdown_mask)
    snl_periods = extract_periods(snl_mask)

    return shutdown_periods, snl_periods

def compute_oee_metrics(df_selected, column_name, shutdown_periods, snl_periods, performance_formula):
    data_timestamp = df_selected[['TimeStamp']].values.flatten()
    sensor_datas = df_selected[column_name].values

    active_power = sensor_datas[:, 0].astype(float)

    nonzeroneg_mask = active_power > 0
    total_hours = (pd.to_datetime(str(data_timestamp[-1])) - pd.to_datetime(str(data_timestamp[0]))).total_seconds() / 3600

    downtime_hours = sum(
        (pd.to_datetime(str(end)) - pd.to_datetime(str(start))).total_seconds() / 3600
        for start, end in shutdown_periods
    )
    snl_hours = sum(
        (pd.to_datetime(str(end)) - pd.to_datetime(str(start))).total_seconds() / 3600
        for start, end in snl_periods
    )

    phy_avail = max(round((total_hours - downtime_hours) / total_hours, 2), 0.01)
    uo_Avail = max(round((total_hours - snl_hours) / total_hours, 2), 0.01)

    if np.any(nonzeroneg_mask):
        log_mean = np.mean(np.log(active_power[nonzeroneg_mask]))
        performance = max(round((performance_formula[0] * log_mean + performance_formula[1]) / 100, 2), 0)
    else:
        performance = 0.01

    oee = max(round(phy_avail * performance * uo_Avail, 2), 0.01)
    datetime_nowMidnight = pd.to_datetime(str(data_timestamp[-1])).replace(hour=1, minute=0, second=0)

    return datetime_nowMidnight, oee, phy_avail, performance, uo_Avail