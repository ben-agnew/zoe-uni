@echo off
REM setup.bat - Setup script for Enron Email Classifier (Windows version)

echo Setting up environment for Enron Email Classifier...

REM Check if Python is installed
python --version > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python and try again.
    exit /b 1
)

REM Pre-install setuptools to avoid distutils error
echo Installing setuptools to avoid potential distutils errors...
python -m pip install setuptools

REM Create virtual environment if it doesn't exist
if not exist env\ (
    echo Creating virtual environment...
    
    REM Try to create virtual environment
    python -m venv env > nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo Error creating virtual environment.
        echo This might be due to missing 'distutils' module.
        echo Trying to install required packages...
        
        echo Installing setuptools via pip...
        python -m pip install setuptools
        
        REM Try again
        echo Retrying virtual environment creation...
        python -m venv env
        if %ERRORLEVEL% NEQ 0 (
            echo Failed to create virtual environment.
            echo Please try the following manually:
            echo 1. python -m pip install setuptools
            echo 2. Or install Visual Studio Build Tools with Python development workload
            pause
            exit /b 1
        )
    )
    
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call env\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

echo Setup complete! Environment is ready.
echo You can now run your classifier with: .\run.bat
echo To deactivate the environment, run: deactivate

echo To start using the environment, run: env\Scripts\activate.bat
echo To run the classifier, run: python main.py

REM Keep console window open
pause
