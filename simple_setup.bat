@echo off
echo Setting up Simple Naive Bayes Email Classifier...
echo ==================================================

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo Python found

REM Create virtual environment if it doesn't exist
if not exist "simple_env" (
    echo Creating virtual environment...
    python -m venv simple_env
    echo Virtual environment created
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment and install packages
echo Activating virtual environment...
call simple_env\Scripts\activate.bat

echo Installing required packages...
python -m pip install --upgrade pip
pip install -r simple_requirements.txt

echo.
echo Setup completed successfully!
echo.
echo To run the simple classifier:
echo 1. Run: simple_run.bat
echo.
pause
