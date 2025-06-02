@echo off
echo Running Simple Naive Bayes Email Classifier...
echo ==============================================

REM Check if virtual environment exists
if not exist "simple_env" (
    echo Error: Virtual environment not found. Please run simple_setup.bat first.
    pause
    exit /b 1
)

REM Check if maildir exists
if not exist "maildir" (
    echo Error: maildir folder not found in current directory.
    echo Please ensure the Enron email dataset 'maildir' folder is present.
    pause
    exit /b 1
)

echo Environment found
echo Dataset found
echo.

REM Activate virtual environment and run script
call simple_env\Scripts\activate.bat
python simple_naive_bayes.py

echo.
pause
