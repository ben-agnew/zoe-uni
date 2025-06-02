#!/bin/bash

# Simple Naive Bayes Email Classifier Run Script
# For macOS/Linux

echo "Running Simple Naive Bayes Email Classifier..."
echo "=============================================="

# Check if virtual environment exists
if [ ! -d "simple_env" ]; then
    echo "❌ Virtual environment not found. Please run simple_setup.sh first."
    exit 1
fi

# Activate virtual environment
source simple_env/bin/activate

# Check if maildir exists
if [ ! -d "maildir" ]; then
    echo "❌ Error: maildir folder not found in current directory."
    echo "Please ensure the Enron email dataset 'maildir' folder is present."
    exit 1
fi

echo "✓ Environment activated"
echo "✓ Dataset found"
echo ""

# Run the script
python simple_naive_bayes.py
