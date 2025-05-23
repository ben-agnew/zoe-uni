#!/bin/zsh
# setup.sh - Setup script for Enron Email Classifier

# Colors for better terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "${YELLOW}Setting up environment for Enron Email Classifier...${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "${RED}Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d ' ' -f 2)
echo "${YELLOW}Using Python version: $PYTHON_VERSION${NC}"

# Pre-install setuptools to avoid distutils error
echo "${YELLOW}Installing setuptools to avoid potential distutils errors...${NC}"
pip3 install setuptools

# Create virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    echo "${YELLOW}Creating virtual environment...${NC}"
    
    # Try to create virtual environment
    if ! python3 -m venv env 2>/dev/null; then
        echo "${RED}Error creating virtual environment.${NC}"
        echo "${YELLOW}This might be due to missing 'distutils' module.${NC}"
        echo "Trying to install required packages..."
        
        # Check if homebrew is installed
        if command -v brew &> /dev/null; then
            echo "Installing python-distutils via Homebrew..."
            brew install python-distutils
        else
            echo "Installing setuptools via pip..."
            pip3 install setuptools
        fi
        
        # Try again
        echo "${YELLOW}Retrying virtual environment creation...${NC}"
        if ! python3 -m venv env; then
            echo "${RED}Failed to create virtual environment.${NC}"
            echo "Please try the following manually:"
            echo "1. pip3 install setuptools"
            echo "2. Or if you use Homebrew: brew install python-distutils"
            exit 1
        fi
    fi
    
    echo "${GREEN}Virtual environment created.${NC}"
else
    echo "${GREEN}Virtual environment already exists.${NC}"
fi

# Activate virtual environment
echo "${YELLOW}Activating virtual environment...${NC}"
source env/bin/activate

# Upgrade pip and setuptools first
echo "${YELLOW}Upgrading pip and setuptools...${NC}"
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

echo "${GREEN}Setup complete! Environment is ready.${NC}"
echo "You can now run your classifier with: ${YELLOW}./run.sh${NC}"
echo "To deactivate the virtual environment, run: ${YELLOW}deactivate${NC}"

echo "To start using the environment, run: ${YELLOW}source env/bin/activate${NC}"
echo "To run the classifier, run: ${YELLOW}python main.py${NC}"
