#!/bin/zsh
# run.sh - Run script for Enron Email Classifier

# Check if virtual environment exists
if [ ! -d "env" ]; then
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source env/bin/activate

# Run the classifier
python main.py
