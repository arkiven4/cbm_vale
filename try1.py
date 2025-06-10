import pickle
import os
import pandas as pd
import numpy as np
import sqlite3
from tqdm import tqdm
import copy
import time
from datetime import datetime, timedelta

import sys
import clr

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

feature_set = ['Active Power', 'Reactive Power', 'Governor speed actual', 'UGB X displacement', 'UGB Y displacement',
    'LGB X displacement', 'LGB Y displacement', 'TGB X displacement',
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

feature_tag_mapping = {
    'Stator winding temperature 13': 'U-LGS1-TI-81104A-AI',
    'Stator winding temperature 14': 'U-LGS1-TI-81104B-AI',
    'Stator winding temperature 15': 'U-LGS1-TI-81104C-AI',
    'Surface Air Cooler Air Outlet Temperature': 'U-LGS1-TI-81104D-AI',
    'Surface Air Cooler Water Inlet Temperature': 'U-LGS1-TI-81104E-AI',
    'Surface Air Cooler Water Outlet Temperature': 'U-LGS1-TI-81104F-AI',
    'Stator core temperature': 'U-LGS1-TI-81104G-AI',
    'UGB metal temperature': 'U-LGS1-TI-81104H-AI',
    'UGB oil temperature': 'U-LGS1-TI-81104I-AI',
    'LGB metal temperature 1': 'U-LGS1-TI-81104J-AI',
    'LGB metal temperature 2': 'U-LGS1-TI-81104K-AI',
    'LGB oil temperature': 'U-LGS1-TI-81104L-AI',
    'Governor speed actual': 'U-LGS1-SI-81101-AI',
    'UGB X displacement': 'U-LGS1-UGB-X-PK-PK-70-AI',
    'UGB Y displacement': 'U-LGS1-UGB-Y-PK-PK-340-AI',
    'LGB X displacement': 'U-LGS1-GB-X-PK-PK-70-AI',
    'LGB Y displacement': 'U-LGS1-LGB-Y-PK-PK-340-AI',
    'TGB X displacement': 'U-LGS1-TGB-X-PK-PK-270-AI',
    'TGB Y displacement': 'U-LGS1-TGB-Y-PK-PK-340-AI',
    'Active Power': 'U-LGS1-Active-Power-AI',
    'Reactive Power': 'U-LGS1-Reactive-Power-AI',
    'Grid Selection': 'U-LGS1-N75-15-0-AI',
    'Opening Wicked Gate': 'U-LGS1-ZT-81101-AI',
    'UGB Oil Contaminant': 'U-LGS1-AY-81103B-AI',
    'Gen Thrust Bearing Oil Contaminant': 'U-LGS1-AY-81103C-AI',
    'Gen Voltage Phase 1': 'U-LGS1-EI_81151A_MV-AI',
    'Gen Voltage Phase 2': 'U-LGS1-EI_81151B_MV-AI',
    'Gen Voltage Phase 3': 'U-LGS1-EI_81151C_MV-AI',
    'Gen Current Phase 1': 'U-LGS1-II_81152A_MV-AI',
    'Gen Current Phase 2': 'U-LGS1-II_81152B_MV-AI',
    'Gen Current Phase 3': 'U-LGS1-II_81152C_MV-AI',
    'Penstock Flow': 'U-LGS1-FI-81101-AI',
    'Turbine flow': 'U-LGS1-FIT-431-AI',
    'UGB cooling water flow': 'U-LGS1-FIT-81103A-AI',
    'LGB cooling water flow': 'U-LGS1-FIT-81103B-AI',
    'Generator cooling water flow': 'U-LGS1-FIT-81103C-AI',
    'Governor Penstock Pressure': 'U-LGS1-PI-81101-AI',
    'Penstock pressure': 'U-LGS1-PT-81150-AI'
}

#pi_tag = [feature_tag_mapping[feature] for feature in feature_set + ['Grid Selection']]
pi_tag = ["U-LGS3_Active-Power-AI","U-LGS3-N75-35-0-AI","U-LGS3_Reactive-Power-AI","U-LGS3_ET_81302-AI","U-LGS3_IT_81302-AI","U-LGS3_SI_81304_I_Eng-AI","U-LGS3_PF_81304_I_Eng-AI","U-LGS3_EI_81304A_I_Eng-AI","U-LGS3_EI_81304B_I_Eng-AI","U-LGS3_EI_81304C_I_Eng-AI","U-LGS3_II_81304A_I_Eng-AI","U-LGS3_II_81304B_I_Eng-AI","U-LGS3_II_81304D_I_Eng-AI","U-LGS3-JI-81104B-AI","U-LGS3-JI-81104C-AI","U-LGS3_SI_81301_I_Eng-AI","U-LGS3_ZT_81301-AI","U-LGS3_PI_81301-AI","U-LGS3-PT-81150-AI","U-LGS3_FI_81301-AI","U-LGS3_FIT_433-AI","U-LGS3-Efficiency-AI","U-LGS-TailRaceLevel-AI","U-LGS3-UGB-X-PK-PK-70-AI","U-LGS3-UGB-Y-PK-PK-340-AI","U-LGS3-GB-X-PK-PK-70-AI","U-LGS3-LGB-Y-PK-PK-340-AI","U-LGS3-TGB-X-PK-PK-270-AI","U-LGS3-TGB-Y-PK-PK-340-AI","U-LGS3_TI_81304H-AI","U-LGS3_TI_81304I-AI","U-LGS3_TI_81304J-AI","U-LGS3_TI_81304K-AI","U-LGS3_TI_81304L-AI","U-LAR3-TT_813UGBWI-AI","U-LAR3-TT_813UGBWO-AI","U-LAR3-TT_813LTBCWO-AI","U-LGS3-FIT-81103C-DI","U-LGS3-FIT-81103A-DI","U-LGS3-FIT-81103B-DI","U-LGS3_TI_81304D-AI","U-LGS3_TI_81304E-AI","U-LGS3_TI_81304F-AI","U-LGS3_TI_81304A-AI","U-LGS3_TI_81304B-AI","U-LGS3_TI_81304C-AI","U-LGS3_TI_81304G-AI","U-LAR3-TT_813SAC1AI-AI","U-LAR3-TT_813SAC2AI-AI","U-LAR3-TT_813SAC3AI-AI","U-LAR3-TT_813SAC4AI-AI","U-LAR3-813SAC1AO-AI","U-LAR3-TT_813SAC2AO-AI","U-LAR3-TT_813SAC4AO-AI","U-LAR3-TT_813SAC1WS-AI","U-LAR3-TT_813SAC2WS-AI","U-LAR3-TT_813SAC4WS-AI","U-LAR3-WDG_TEMP_PHASE_A1.In-AI","U-LAR3-WDG_TEMP_PHASE_A2.In-AI","U-LAR3-WDG_TEMP_PHASE_A3.In-AI","U-LAR3-WDG_TEMP_PHASE_B1.In-AI","U-LAR3-WDG_TEMP_PHASE_B2.In-AI","U-LAR3-WDG_TEMP_PHASE_B3.In-AI","U-LAR3-WDG_TEMP_PHASE_C1.In-AI","U-LAR3-WDG_TEMP_PHASE_C2.In-AI","U-LAR3-WDG_TEMP_PHASE_C3.In-AI","U-LAR3-TT_813SC2-AI","U-LAR3-TT_813SC3-AI","U-LAR3-TT_813SC4-AI","U-LAR3-TT_813SC5-AI","U-LAR3-TT_813SC6-AI","U-LAR3-TT_813SC7-AI","U-LAR3-TT_813SC8-AI","U-LAR3-TT_813SC9-AI","U-LAR3-TT_813SC10-AI","U-LGS3-AY-81103B-DI","U-LGS3-AY-81103C-DI","U-LGS3_BPSP_81301B-AI","U-LGS3_BPVI_81301C-AI","U-LGS3_ZSP_81301C-AI","U-LGS3_SSP_81301B-AI","U-LGS3-JSP-81101A-AI","U-LGS3_JSP_81301B-AI","U-LGS3-ACTIVE_PWR_SETPOINT-AI","U-LGS3-LAR-MVAR-SP-AI","U-LGS3-ZI-81102A-AI","U-LGS3-ZI-81102B-AI"]

piServers = PIServers()
piServer = piServers["PTI-PI"]                                                    #Write PI Server Name
piServer.Connect(False)                                                             #Connect to PI Server
print ('Connected to server: ' + "PTI-PI")

time_list = ['2020-01-01 00:00:00', '2020-12-28 09:38:17']
timerange = AFTimeRange(time_list[0], time_list[1])
master_pd = ""
for i in range(22, len(pi_tag)):
    try:
        print(pi_tag[i] + "_" + str(i))
        tag = PIPoint.FindPIPoint(piServer, pi_tag[i])
        value_resp = parse_recorded_events(tag.InterpolatedValues(timerange, AFTimeSpan.Parse('1m'), '', False))
        value_resp['Timestamps'] = pd.to_datetime(value_resp['Timestamps'])
        master_pd = value_resp
        master_pd.to_csv("save_csv2/" + pi_tag[i] + '.csv', index=False)
    except Exception as e:
        try:
            print(f"Unexpected error for tag {pi_tag[i]}: {e}")
            time.sleep(25 * 60)
            tag = PIPoint.FindPIPoint(piServer, pi_tag[i])
            value_resp = parse_recorded_events(tag.InterpolatedValues(timerange, AFTimeSpan.Parse('1m'), '', False))
            value_resp['Timestamps'] = pd.to_datetime(value_resp['Timestamps'])
            master_pd = value_resp
            master_pd.to_csv("save_csv2/" + pi_tag[i] + '.csv', index=False)
        except:
            print(f"Still Error for tag {pi_tag[i]}: {e}")