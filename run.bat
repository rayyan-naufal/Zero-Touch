@echo off
title Zero Touch - Windows Touchless Head & Hand Mouse
cd /d "%~dp0"
echo Starting Zero Touch...
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with an error.
    pause
)
