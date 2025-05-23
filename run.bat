@echo off
REM run.bat - Run script for Enron Email Classifier (Windows version)

REM Check if virtual environment exists
if not exist env\ (
    echo Virtual environment not found. Please run setup.bat first.
    exit /b 1
)

REM Activate virtual environment
call env\Scripts\activate.bat

REM Run the classifier
python main.py

pause
