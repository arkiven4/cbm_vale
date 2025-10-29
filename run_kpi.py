import sys
sys.path.extend(['./accumulation_tree', './tdigest'])
from accumulation_tree import AccumulationTree
from tdigest import TDigest

import os, pickle, sqlite3, copy, time, sklearn, sys, clr
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from scipy.signal import resample

import src.commons as commons
import src.custom_const as custom_const

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

def timeseries_savedb_many(records, db_name="data.db", table_name="sensor_data"):
    """
    records: list of tuples in format:
        (datetime_obj, data_array, feature_set_list)
    db_name: SQLite database path
    table_name: table to insert into

    The table must have UNIQUE(timestamp) for REPLACE to work.
    """
    if not records:
        return

    # All records should have the same feature set
    feature_set = records[0][2]
    feature_columns = ', '.join([f.replace(" ", "_") for f in feature_set])
    placeholders = ', '.join(['?' for _ in range(len(feature_set))])

    sql = f"""
        INSERT OR REPLACE INTO {table_name} (timestamp, {feature_columns})
        VALUES (?, {placeholders})
    """

    # Prepare all values for executemany
    values = []
    for ts, data, _ in records:
        ts_str = ts
        values.append((ts_str, *data))

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.executemany(sql, values)
    conn.commit()
    conn.close()

def parse_recorded_events(recorded):
    parsed_events = []
    for event in recorded:
        # Convert .NET DateTime to string
        timestamp_str = str(event.Timestamp.LocalTime)
        # Convert to pandas Timestamp
        timestamp = pd.to_datetime(timestamp_str)
        value = event.Value
        parsed_events.append((timestamp, value))
    return pd.DataFrame(parsed_events, columns=['Timestamps', 'Values'])


def getdf_piserverKPI(piServer, pi_tag, time_list, feature_set):
    timerange = AFTimeRange(time_list[0], time_list[1])
    master_pd = ""
    for i in range(len(pi_tag)):
        tag = PIPoint.FindPIPoint(piServer, pi_tag[i])
        value_resp = parse_recorded_events(tag.InterpolatedValues(
            timerange, AFTimeSpan.Parse('1m'), '', False))
        if i == 0:
            value_resp['Timestamps'] = pd.to_datetime(value_resp['Timestamps'])
            master_pd = value_resp
        else:
            master_pd = pd.concat(
                [master_pd, value_resp['Values']], axis=1, join='inner')

    master_pd = master_pd.values
    master_pd = pd.DataFrame(data=master_pd, columns=['TimeStamp'] + feature_set)
    # master_pd.replace('I/O Timeout', np.nan, inplace=True)
    # master_pd.replace('No Data', np.nan, inplace=True)
    # master_pd.replace('Future Data Unsupported', np.nan, inplace=True)
    # master_pd.replace('Closed', np.nan, inplace=True)
    # master_pd.replace('Open', np.nan, inplace=True)
    for column_name in master_pd.columns:
        if column_name != 'Load_Type' and column_name != 'TimeStamp':
            master_pd[column_name] = pd.to_numeric(
                master_pd[column_name], errors='coerce', downcast='float')
    master_pd = master_pd.sort_values(by='TimeStamp')
    master_pd = master_pd.reset_index(drop=True)

    #master_pd = master_pd.mask(master_pd < 0).fillna(method='ffill')
    numeric_cols = master_pd.select_dtypes(include=[np.number]).columns
    master_pd[numeric_cols] = master_pd[numeric_cols].mask(master_pd[numeric_cols] < 0).fillna(method='ffill')
    master_pd = master_pd.fillna(method='ffill')
    master_pd['Total Karebbe Power Daily (Tot)'] = master_pd['Total Balambano Power Daily (Tot)'] # Temp Fix

    df_sel = master_pd.reset_index(drop=True)
    df_sel = df_sel[['TimeStamp'] + feature_set]
    return df_sel


############################ Configuration ###############################
interval_gap = 10 * 60  # Seconds

plant_metadata = {
    'Larona': [{
        'name': "LGS1",
        'active_power': 'LGS1 Active Power',
        'rpm': 'LGS1 Governor Unit Speed Actual',
        'aux': 'LGS1-Auxiliary Grid (0 = ACTIVE)',
        'coef': [20.944, 11.398]
    },
        {
        'name': "LGS2",
        'active_power': 'LGS2 Active Power',
        'rpm': 'LGS2 Governor Unit Speed Actual',
        'aux': 'LGS2-Auxiliary Grid (0 = ACTIVE)',
        'coef': [21.162, 8.49]
    },
        {
        'name': "LGS3",
        'active_power': 'LGS3 Active Power',
        'rpm': 'LGS3 Governor Unit Speed Actual',
        'aux': 'LGS3-Auxiliary Grid (0 = ACTIVE)',
        'coef': [19.66, 13.676]
    }],
    'Balambano': [{
        'name': "BGS1",
        'active_power': 'BGS1 Power',
        'rpm': 'GEN SPEED BGS1',
        'aux': 'BGS1-Auxiliary Grid (0 = ACTIVE)',
        'coef': [17.44, 6.87]
    },
        {
        'name': "BGS2",
        'active_power': 'BGS2 Power',
        'rpm': 'GEN SPEED BGS2',
        'aux': 'BGS2-Auxiliary Grid (0 = ACTIVE)',
        'coef': [21.11, -1.25]
    }],
    'Karebbe': [{
        'name': "KGS1",
        'active_power': 'K U1 Active Power (MW)',
        'rpm': 'K U1 Turb Gov Turbine Speed (RPM)',
        'aux': 'KGS1-Auxiliary Grid (0 = ACTIVE)',
        'coef': [19.64, 11.05]
    },
        {
        'name': "KGS2",
        'active_power': 'K U2 Active Power (MW)',
        'rpm': 'K U2 Turb Gov Turbine Speed (RPM)',
        'aux': 'KGS2-Auxiliary Grid (0 = ACTIVE)',
        'coef': [16.84, 23.41]
    }]
}

############################ Setup ###############################
for value in plant_metadata.values():
    for value2 in value:
        commons.init_db_timeconst(['oee', 'phy_avail', 'performance', 'uo_Avail', "aux_0", "aux_1"], "db/kpi.db", value2['name'])
        commons.init_db_timeconst(['active_power', 'rpm', "aux_0", "aux_1"], "db/kpi.db", value2['name'] + "_timeline")

commons.init_db_timeconst(['hpd', 'ahpa', 'lpd', 'bpd', 'kpd'], "db/kpi.db", "PowerProd")

############################ Connect PI Server ####################
piServers = PIServers()
piServer = piServers["PTI-PI"]
piServer.Connect(False)
print('Connected to server: ' + "PTI-PI")

############################ Main Loops  ##########################
count = 0
last_execution_date_kpi = None
while True:
    now_time = datetime.utcnow()
    if (last_execution_date_kpi != now_time.date() and now_time.hour >= 1):
        count = count + 1
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Executing KPI Task #{count}...")

        today = now_time.date()
        start_time = now_time - timedelta(hours=24)
        time_list = [start_time.strftime(
            '%Y-%m-%d %H:%M:%S'), now_time.strftime('%Y-%m-%d %H:%M:%S')]

        df_selkpi = getdf_piserverKPI(piServer, [v for k, v in custom_const.feature_tag_mappingKPI.items()],
                                      time_list, [k for k, v in custom_const.feature_tag_mappingKPI.items()])
        power_prod_df = []
        for value in plant_metadata.values():
            for tags in value:
                unit_name = tags['name']
                if tags['active_power'] not in df_selkpi.columns or tags['rpm'] not in df_selkpi.columns:
                    continue

                df_unit = df_selkpi[[
                    'TimeStamp', tags['active_power'], tags['rpm'], tags['aux']]].dropna()
                if df_unit.empty:
                    continue
                
                df_unit_powerprod = df_unit.copy()
                df_unit_powerprod = df_unit_powerprod.drop([tags['rpm'], tags['aux']], axis=1)
                df_unit_powerprod = df_unit_powerprod.rename(columns={tags['active_power']: "Active Power"})
                df_unit_powerprod["Unit"] = unit_name
                power_prod_df.append(df_unit_powerprod)

                # Process shutdown & SNL
                shutdown_periods, snl_periods = commons.process_shutdown_and_snl_periods(
                    df_unit, [tags['active_power'], tags['rpm']]
                )

                # Compute OEE and related KPIs
                datetime_nowMidnight, oee, phy_avail, performance, uo_Avail = commons.compute_oee_metrics(
                    df_unit, [tags['active_power'], tags['rpm']],
                    shutdown_periods, snl_periods,
                    performance_formula=tags['coef']
                )

                # Count Auxiliary Grid ON/OFF
                counts_aux = df_unit[tags['aux']].value_counts().sort_index()
                aux_0 = counts_aux.get(0.0, 0)
                aux_1 = counts_aux.get(1.0, 0)

                # Save to database
                commons.timeseries_savedb(
                    datetime_nowMidnight,
                    np.array([oee, phy_avail, performance,
                             uo_Avail, aux_0, aux_1]),
                    ['oee', 'phy_avail', 'performance',
                        'uo_Avail', 'aux_0', 'aux_1'],
                    "db/kpi.db",
                    unit_name
                )

        power_prod_df = pd.concat(power_prod_df, ignore_index=True)
        power_prod_df["TimeStamp"] = pd.to_datetime(power_prod_df["TimeStamp"])
        hourly = (
            power_prod_df.set_index("TimeStamp")
                .groupby("Unit")["Active Power"]
                .resample("H").mean()
                .reset_index()
        )
        daily = (
            hourly.set_index("TimeStamp")
                .groupby("Unit")["Active Power"]
                .resample("D").sum()
                .reset_index()
        )
        daily["Group"] = daily["Unit"].str[0]
        final = (
            daily.groupby(["Group","TimeStamp"])["Active Power"]
                .sum()
                .reset_index()
                .pivot(index="TimeStamp", columns="Group", values="Active Power")
                .reset_index()
        )
        final.columns.name = None
        final.columns = ["Date", "LGS", "BGS", "KGS"] # Total, AVG
        final["Total Hydro Power"] = final[["LGS","BGS","KGS"]].sum(axis=1)
        final["Avg Hydro Power"] = final[["LGS","BGS","KGS"]].mean(axis=1)
        
        pda_datas = final.values[-1]
        commons.timeseries_savedb(
            datetime_nowMidnight,
            np.array([pda_datas[4], pda_datas[5], pda_datas[1],
                     pda_datas[2], pda_datas[3]]).astype(np.float64),
            ['hpd', 'ahpa', 'lpd', 'bpd', 'kpd'],
            "db/kpi.db",
            "PowerProd"
        )

        trim_len = (len(df_selkpi) // 15) * 15
        timestamps = df_selkpi['TimeStamp'].iloc[:trim_len:15].reset_index(drop=True)
        vals = df_selkpi.select_dtypes(include=np.number).values[:trim_len]
        means = vals.reshape(-1, 15, vals.shape[1]).mean(axis=1)
        df_selkpi_15min = pd.DataFrame(means, columns=df_selkpi.select_dtypes(include=np.number).columns)
        df_selkpi_15min['TimeStamp'] = timestamps

        for value in plant_metadata.values():
            for tags in value:
                unit_name = tags['name']

                # skip if required columns missing
                required_cols = [tags['active_power'], tags['rpm'], tags['aux']]
                if not all(c in df_selkpi_15min.columns for c in required_cols):
                    continue

                df_unit = df_selkpi_15min[['TimeStamp'] + required_cols].fillna(0)

                # process all rows at once
                binary_vals = (df_unit[tags['aux']] >= 0.5).astype(int)
                aux_0 = (binary_vals == 0).astype(int)
                aux_1 = (binary_vals == 1).astype(int)

                # stack TimeStamp, active_power, rpm, aux_0, aux_1 into 2D array
                unit_records = np.column_stack([
                    df_unit['TimeStamp'].astype(str).values,
                    df_unit[tags['active_power']].values,
                    df_unit[tags['rpm']].values,
                    aux_0.values,
                    aux_1.values
                ])

                # prepare records to insert for this unit only
                records_to_insert = [
                    (r[0], np.array(r[1:], dtype=float), ['active_power', 'rpm', 'aux_0', 'aux_1'])
                    for r in unit_records
                ]

                timeseries_savedb_many(records_to_insert, db_name="db/kpi.db", table_name=unit_name + "_timeline")
                    
        # DONT REMOVE THIS
        last_execution_date_kpi = today

    # DONT REMOVE THIS
    time.sleep(interval_gap)
