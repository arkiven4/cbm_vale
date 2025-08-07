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

feature_tag_mappingKPI = {
    'Avg Hydro Power Available 1D (Avg)': 'U-PWR-HYDRO-AI-AVGD',
    'Total Hydro Power Daily (Tot)': 'U-HGST-Power-AI-DTT',
    'Total Larona Power Daily (Tot)': 'U-PWR-LAR-TOT-DTT',
    'Total Balambano Power Daily (Tot)': 'U-PWR-BAL-TOT-DTT',
    'Total Karebbe Power Daily (Tot)': 'U-PWR-BAL-TOT-DTT', 

    # LGS
    'LGS1 Active Power': 'U-LGS1-Active-Power-AI',
    'LGS1-Auxiliary Grid (0 = ACTIVE)': 'U-LGS1-N75-15-0-AI',
    'LGS1 Governor Unit Speed Actual': 'U-LGS1-SI-81101-AI',
    
    'LGS2 Active Power': 'U-LGS2-Active-Power-AI',
    'LGS2-Auxiliary Grid (0 = ACTIVE)': 'U-LGS2-N75-25-0-AI',
    'LGS2 Governor Unit Speed Actual': 'U-LGS2-SI-81201-AI',
    
    'LGS3 Active Power': 'U-LGS3_Active-Power-AI',
    'LGS3-Auxiliary Grid (0 = ACTIVE)': 'U-LGS3-N75-35-0-AI',
    'LGS3 Governor Unit Speed Actual': 'U-LGS3_SI_81301_I_Eng-AI',

    # BGS
    'BGS1 Power': 'U-BGS1-Power-AI',
    'BGS1-Auxiliary Grid (0 = ACTIVE)': 'U-BGS1-N75-45-0-AI',
    'GEN SPEED BGS1': 'U-BGS1_I_T_SPEED-AI',

    'BGS2 Power': 'U-BGS2-Power-AI',
    'BGS2-Auxiliary Grid (0 = ACTIVE)': 'U-BGS2-N75-55-0-AI',
    'GEN SPEED BGS2': 'U-BGS2_I_T_SPEED-AI',

    # KGS
    'K U1 Active Power (MW)': 'U-KGS1-Active_Power_AI',
    'KGS1-Auxiliary Grid (0 = ACTIVE)': 'U-KGS1-N75-65-0-AI',
    'K U1 Turb Gov Turbine Speed (RPM)': 'U-KGS1-Turb_Gov_Turb_Speed-AI',

    'K U2 Active Power (MW)': 'U-KGS2-Active_Power_AI',
    'KGS2-Auxiliary Grid (0 = ACTIVE)': 'U-KGS2-N75-75-0-AI',
    'K U2 Turb Gov Turbine Speed (RPM)': 'U-KGS2-Turb_Gov_Turb_Speed-AI',
}