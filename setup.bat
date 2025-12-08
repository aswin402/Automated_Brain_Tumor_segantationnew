@echo off
REM Brain Tumor ML Pipeline Setup for Windows

echo ==================================
echo Brain Tumor ML Pipeline Setup
echo ==================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

echo.
echo Creating virtual environment...
python -m venv venv
echo Virtual environment created
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated
echo.

echo Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
echo Pip upgraded
echo.

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
echo All dependencies installed
echo.

echo ==================================
echo Setup Complete!
echo ==================================
echo.
echo To activate the virtual environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To run the pipeline:
echo   python main.py
echo.
echo To deactivate the virtual environment:
echo   deactivate
echo.
pause
