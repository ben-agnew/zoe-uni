# Enron Email Sender Classifier

This project implements a Naive Bayes Classifier to identify the sender of emails from the Enron email dataset. The classifier is trained on emails from the 150 most prolific senders in the dataset.

## Project Overview

The Enron email dataset contains approximately 500,000 emails from about 150 users, primarily senior management at Enron. This project:

1. Processes emails from the Enron dataset
2. Trains a Naive Bayes Classifier to identify the sender of an email
3. Provides an interactive interface to test the classifier

## Setup Instructions

### Prerequisites

-   Python 3.8 or higher
-   Enron email dataset (`maildir` folder)

### Common Setup Issues

#### Missing distutils Module

If you encounter `ModuleNotFoundError: No module named 'distutils'` (common with Python 3.12+), try one of these solutions:

**For macOS:**

```
# If you use Homebrew
brew install python-distutils

# Or install setuptools
pip3 install --user setuptools
```

**For Ubuntu/Debian:**

```
sudo apt-get install python3-distutils
```

**For Windows:**

```
python -m pip install setuptools
```

The setup scripts have been updated to try to handle this automatically, but manual installation might be needed in some cases.

### Setting up the Environment

#### For macOS/Linux:

1. Run the setup script to create a virtual environment and install dependencies:

    ```
    ./setup.sh
    ```

2. Alternatively, you can set up manually:
    ```
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```

#### For Windows:

1. Run the setup batch file:

    ```
    setup.bat
    ```

2. Alternatively, you can set up manually:
    ```
    python -m venv env
    env\Scripts\activate
    pip install -r requirements.txt
    ```

## Running the Classifier

### For macOS/Linux:

1. Using the provided script:

    ```
    ./run.sh
    ```

2. Or manually:
    ```
    source env/bin/activate
    python main.py
    ```

### For Windows:

1. Using the provided batch file:

    ```
    run.bat
    ```

2. Or manually:
    ```
    env\Scripts\activate
    python main.py
    ```

## Features

-   **Text Processing**: Extracts content from raw email files and performs basic cleaning
-   **Vectorization**: Uses TF-IDF vectorization to convert text to numerical features
-   **Model Training**: Trains a Multinomial Naive Bayes classifier
-   **Hyperparameter Optimization**: Uses GridSearchCV to find the optimal model parameters
-   **Interactive Testing**: Allows testing the model with custom email text
-   **Model Persistence**: Saves trained models for reuse without retraining
-   **Model Metadata**: Tracks model performance and training details

## Project Structure

-   `main.py`: Main script containing the classifier implementation
-   `requirements.txt`: List of Python dependencies
-   `setup.sh`: Script to set up the virtual environment (macOS/Linux)
-   `setup.bat`: Script to set up the virtual environment (Windows)
-   `run.sh`: Script to run the classifier (macOS/Linux)
-   `run.bat`: Script to run the classifier (Windows)
-   `maildir/`: Directory containing the Enron email dataset
-   `processed_emails.csv`: Processed dataset (created after first run)
-   `models/`: Directory containing saved models (created after first run)
    -   `baseline_model.joblib`: Saved baseline Naive Bayes model
    -   `optimized_model.joblib`: Saved hyperparameter-optimized model
    -   `model_metadata.json`: Information about trained models

## Performance

The model typically achieves an accuracy of 70-80% on the test set, depending on the specific emails and users included in the dataset.

## License

This project is provided for educational purposes only.
