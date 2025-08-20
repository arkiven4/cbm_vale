@echo off
cd /d "%~dp0"
python run_cbm.py --dataset CustomAWGN30ES15 --model "" --Device cpu --test
