@echo off
REM Change directory to where your script is located
D:
cd /d "D:\ITSTeam\NewGen\cbm_vale"

REM Run the Python script with arguments
start cmd /k python run_cbm.py --dataset CustomAWGN30ES15 --model "" --Device cpu --test
start cmd /k python run_kpi.py

REM Exit this launcher terminal
exit