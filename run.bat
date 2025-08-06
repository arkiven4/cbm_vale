@echo off
REM Change directory to where your script is located
D:
cd /d "D:\ITSTeam\NewGen\cbm_vale"

REM Run the Python script with arguments
python run.py --dataset CustomAWGN30ES15 --model "" --Device cpu --test

REM Optional: pause to see output before window closes
pause