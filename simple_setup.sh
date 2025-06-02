#!/bin/bash

# Simple Naive Bayes Email Classifier Setup Script
# For macOS/Linux

echo "Setting up Simple Naive Bayes Email Classifier..."
echo "================================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found"

# Create virtual environment if it doesn't exist
if [ ! -d "simple_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv simple_env
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source simple_env/bin/activate

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip
pip install -r simple_requirements.txt

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To run the simple classifier:"
echo "1. Activate the environment: source simple_env/bin/activate"
echo "2. Run the script: python simple_naive_bayes.py"
echo ""
echo "Or use the run script: ./simple_run.sh"
