@echo off
D:
cd /d "D:\ITSTeam\NewGen\cbm_vale-main"
python run.py --dataset CustomAWGN30ES15 --model "" --Device cpu --test
pause