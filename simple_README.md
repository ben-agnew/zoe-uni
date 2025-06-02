# Simple Naive Bayes Email Classifier

A streamlined Python script for email sender classification using the Enron dataset and Naive Bayes classifier.

## Overview

This simplified script focuses specifically on:

1. **Dataset Creation**: Processing Enron emails and creating clean datasets
2. **Data Splitting**: Creating train/validation/test splits with stratification
3. **Naive Bayes Training**: Training both baseline and hyperparameter-optimized models
4. **Model Evaluation**: Comprehensive performance analysis and comparison
5. **Interactive Testing**: Real-time email sender prediction

## Features

-   **Streamlined Pipeline**: End-to-end workflow in a single script
-   **Automatic Dataset Processing**: Extracts and cleans email content from maildir format
-   **Duplicate Handling**: Removes duplicate emails and ensures data quality
-   **Hyperparameter Optimization**: Grid search for optimal TF-IDF and Naive Bayes parameters
-   **Comprehensive Evaluation**: Multiple metrics including accuracy, precision, recall, F1-scores
-   **Model Persistence**: Saves trained models and metadata for reuse
-   **Interactive Testing**: Test the model with custom email text

## Requirements

-   Python 3.8 or higher
-   Enron email dataset (maildir folder)
-   Python packages (see simple_requirements.txt):
    -   scikit-learn
    -   pandas
    -   numpy
    -   joblib

## Setup

1. **Install dependencies:**

    ```bash
    pip install -r simple_requirements.txt
    ```

2. **Ensure dataset is available:**
   Make sure you have the Enron email dataset in a `maildir` folder in the same directory as the script.

## Usage

Simply run the script:

```bash
python simple_naive_bayes.py
```

The script will automatically:

1. Process emails from the maildir folder
2. Create train/validation/test splits
3. Train a baseline Naive Bayes model
4. Optimize hyperparameters using grid search
5. Compare model performance
6. Offer interactive testing

## Output Files

The script creates several output files:

-   `simple_processed_emails.csv` - Cleaned and processed email dataset
-   `simple_train_data.csv` - Training set
-   `simple_validation_data.csv` - Validation set
-   `simple_test_data.csv` - Test set
-   `simple_baseline_model.joblib` - Baseline Naive Bayes model
-   `simple_optimized_model.joblib` - Hyperparameter-optimized model
-   `simple_evaluation_results.json` - Detailed evaluation results and comparison

## Dataset Processing

The script automatically:

-   Extracts content from email files in sent/sent_items/\_sent_mail directories
-   Removes email headers, metadata, and formatting
-   Filters out very short emails (< 50 characters)
-   Removes duplicate emails based on content
-   Keeps only users with at least 20 emails
-   Selects top 150 users by email volume

## Model Training

**Baseline Model:**

-   Default scikit-learn MultinomialNB parameters
-   TF-IDF vectorization with 5000 features
-   Quick training for baseline comparison

**Optimized Model:**

-   Grid search across multiple parameters:
    -   TF-IDF max_features: [3000, 5000, 7000]
    -   TF-IDF min_df: [2, 3, 5]
    -   TF-IDF max_df: [0.7, 0.8, 0.9]
    -   Naive Bayes alpha: [0.01, 0.1, 1.0, 10.0]
-   Validation-based parameter selection
-   108 parameter combinations tested

## Performance Metrics

The script evaluates models using:

-   **Accuracy**: Overall classification accuracy
-   **Precision (weighted)**: Precision averaged by class support
-   **Recall (weighted)**: Recall averaged by class support
-   **F1-Score (weighted)**: Harmonic mean of precision and recall
-   **F1-Score (macro)**: Unweighted average across all classes

## Sample Output

```
SIMPLIFIED NAIVE BAYES EMAIL CLASSIFIER
========================================
Final dataset: 95,573 emails from 144 unique senders

BASELINE MODEL RESULTS:
Test Accuracy: 0.5738
Test F1 (weighted): 0.6315

OPTIMIZED MODEL RESULTS:
Test Accuracy: 0.6721
Test F1 (weighted): 0.6627

📈 Accuracy improvement: +16.9%
```

## Interactive Testing

After training, you can test the model with custom email text:

```
📧 Email text: Thanks for the update. Let me know if you need anything else.
🔮 Prediction: john-doe
📊 Confidence: 0.7854

Top 3 candidates:
  1. john-doe: 0.7854
  2. jane-smith: 0.1205
  3. bob-wilson: 0.0891
```

## Advantages of This Simplified Version

-   **Focused**: Only Naive Bayes classifier (no complex model comparisons)
-   **Fast**: Optimized for quick iteration and testing
-   **Self-contained**: Single file with all functionality
-   **Educational**: Clear, readable code structure
-   **Practical**: Includes all essential steps from data to deployment

## Comparison with Main Script

| Feature        | Main Script         | Simple Script       |
| -------------- | ------------------- | ------------------- |
| Models         | 6 algorithms        | Naive Bayes only    |
| Complexity     | ~2500 lines         | ~500 lines          |
| Training time  | 10-30 minutes       | 2-5 minutes         |
| Visualizations | Extensive plots     | Text-based results  |
| Focus          | Research comparison | Production pipeline |

This simplified version is perfect for:

-   Learning email classification fundamentals
-   Quick prototyping and testing
-   Production deployments where only Naive Bayes is needed
-   Educational purposes and demonstrations
