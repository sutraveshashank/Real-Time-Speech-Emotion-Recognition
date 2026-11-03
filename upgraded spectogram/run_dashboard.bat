@echo off
title SoundSense AI Dashboard Launcher
echo ========================================================
echo   Launching SoundSense AI Web Dashboard...
echo ========================================================
cd /d "%~dp0"
streamlit run app.py
pause

