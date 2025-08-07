import os, pickle, sqlite3, copy, time, sklearn, sys, clr
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
from scipy.signal import resample

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

def parse_recorded_events(recorded):
    parsed_events = []
    for event in recorded:
        timestamp_str = str(event.Timestamp.LocalTime)  # Convert .NET DateTime to string
        timestamp = pd.to_datetime(timestamp_str)  # Convert to pandas Timestamp
        value = event.Value
        parsed_events.append((timestamp, value))
    return pd.DataFrame(parsed_events, columns=['Timestamps', 'Values'])


def getdf_piserverKPI(piServer, pi_tag, time_list, feature_set):
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
    master_pd = pd.DataFrame(data=master_pd, columns=['TimeStamp'] + feature_set)
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
    
    df_sel = master_pd.reset_index(drop=True)
    df_sel = df_sel[['TimeStamp'] + feature_set] 
    return df_sel

############################ Configuration ###############################
interval_gap = 10 * 60 # Seconds

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
        'coef': [20.944, 11.398]
    },
    {
        'name': "BGS2",
        'active_power': 'BGS2 Power',
        'rpm': 'GEN SPEED BGS2',
        'aux': 'BGS2-Auxiliary Grid (0 = ACTIVE)',
        'coef': [21.162, 8.49]
    }],
    'Karebbe': [{
        'name': "KGS1",
        'active_power': 'K U1 Active Power (MW)',
        'rpm': 'K U1 Turb Gov Turbine Speed (RPM)',
        'aux': 'KGS1-Auxiliary Grid (0 = ACTIVE)',
        'coef': [20.944, 11.398]
    },
    {
        'name': "KGS2",
        'active_power': 'K U2 Active Power (MW)',
        'rpm': 'K U2 Turb Gov Turbine Speed (RPM)',
        'aux': 'KGS2-Auxiliary Grid (0 = ACTIVE)',
        'coef': [21.162, 8.49]
    }]
}

############################ Setup ###############################
for value in plant_metadata.values():
    for value2 in value:
        commons.init_db_timeconst(['oee', 'phy_avail', 'performance', 'uo_Avail', "aux_0", "aux_1"], "db/kpi.db", value2['name'])
commons.init_db_timeconst(['hpd', 'ahpa', 'lpd', 'bpd', 'kpd'], "db/kpi.db", "PowerProd")

############################ Connect PI Server ####################
piServers = PIServers()
piServer = piServers["PTI-PI"]
piServer.Connect(False) 
print ('Connected to server: ' + "PTI-PI")

############################ Main Loops  ##########################
count = 0
last_execution_date_kpi = None
while True:
    now_time = datetime.utcnow()
    if now_time.hour >= 1 and (last_execution_date_kpi is None or last_execution_date_kpi < now_time.date()):
        count = count + 1
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Executing KPI Task #{count}...")

        today = now_time.date()
        start_time = now_time - timedelta(hours=24)
        time_list = [start_time.strftime('%Y-%m-%d %H:%M:%S'), now_time.strftime('%Y-%m-%d %H:%M:%S')]
        
        df_selkpi = getdf_piserverKPI(piServer, [v for k,v in custom_const.feature_tag_mappingKPI.items()], 
                                      time_list, [k for k,v in custom_const.feature_tag_mappingKPI.items()])
        for value in plant_metadata.values():
            for tags in value:
                unit_name = tags['name']
                if tags['active_power'] not in df_selkpi.columns or tags['rpm'] not in df_selkpi.columns:
                    continue

                df_unit = df_selkpi[['TimeStamp', tags['active_power'], tags['rpm'], tags['aux']]].dropna()
                if df_unit.empty:
                    continue

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
                timeseries_savedb(
                    datetime_nowMidnight,
                    np.array([oee, phy_avail, performance, uo_Avail, aux_0, aux_1]),
                    ['oee', 'phy_avail', 'performance', 'uo_Avail', 'aux_0', 'aux_1'],
                    "db/kpi.db",
                    unit_name
                )

        pda_datas = df_selkpi[['Total Hydro Power Daily (Tot)', 'Avg Hydro Power Available 1D (Avg)' , 'Total Larona Power Daily (Tot)', 'Total Balambano Power Daily (Tot)', 'Total Karebbe Power Daily (Tot)']].mean().values
        timeseries_savedb(
                datetime_nowMidnight,
                np.array([pda_datas[0], pda_datas[1], pda_datas[2], pda_datas[3], pda_datas[3]]).astype(np.float64),
                ['hpd', 'ahpa', 'lpd', 'bpd', 'kpd'],
                "db/kpi.db",
                "PowerProd"
            )

        # DONT REMOVE THIS
        last_execution_date_kpi = today

    # DONT REMOVE THIS
    df_timestamp_last = df_timestamp[-1]
    time.sleep(interval_gap)