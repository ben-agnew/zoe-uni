"""
Naive Bayes Classifier for Enron Email Dataset
This script uses the Enron email dataset to classify emails by sender.

Setup instructions:
For macOS/Linux:
1. Create a virtual environment: python -m venv env
2. Activate the environment: source env/bin/activate
3. Install dependencies: pip install -r requirements.txt
4. Run the script: python main.py
   
For Windows:
1. Create a virtual environment: python -m venv env
2. Activate the environment: env\Scripts\activate
3. Install dependencies: pip install -r requirements.txt
4. Run the script: python main.py

Or simply:
- On macOS/Linux: run ./setup.sh and then ./run.sh
- On Windows: run setup.bat and then run.bat
"""

import datetime
import email
import json
import os
import re
import time

import joblib  # For saving/loading models
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Matplotlib and/or Seaborn not available. Visualizations will be skipped.")
    print("To enable visualizations, install the required packages:")
    print("pip install matplotlib seaborn")

# Constants
DATA_DIR = "enron_data"
PROCESSED_DATA_FILE = "processed_emails.csv"
TRAIN_DATA_FILE = "train_data.csv"
TEST_DATA_FILE = "test_data.csv"
NUM_TOP_USERS = 150  # Number of unique users to include
MODEL_DIR = "models"  # Directory to store models
RESULTS_DIR = "results"  # Directory to store evaluation results
BASELINE_MODEL_FILE = os.path.join(MODEL_DIR, "baseline_model.joblib")
OPTIMIZED_MODEL_FILE = os.path.join(MODEL_DIR, "optimized_model.joblib")
MODEL_METADATA_FILE = os.path.join(MODEL_DIR, "model_metadata.json")

def extract_email_content(email_path):
    """Extract the content from an email file."""
    try:
        with open(email_path, 'r', encoding='latin1') as f:
            msg = email.message_from_file(f)
            
        subject = msg.get('Subject', '')
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    payload = part.get_payload(decode=True)
                    if payload:
                        body += payload.decode('latin1', errors='ignore')
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode('latin1', errors='ignore')
                
        # Combine subject and body
        content = f"{subject} {body}"
        
        # Basic cleaning
        content = re.sub(r'\s+', ' ', content)  # Remove extra whitespace
        content = re.sub(r'[^\w\s]', '', content)  # Remove punctuation
        content = content.lower()  # Convert to lowercase
        
        return content
    except Exception as e:
        print(f"Error processing {email_path}: {e}")
        return ""

def process_emails(maildir_path):
    """Process all emails and create a DataFrame."""
    if os.path.exists(PROCESSED_DATA_FILE):
        print(f"Loading processed data from {PROCESSED_DATA_FILE}")
        return pd.read_csv(PROCESSED_DATA_FILE)
    
    print(f"Processing emails from existing maildir at {maildir_path}...")
    emails_data = []
    
    # Walk through the maildir directory
    for user in os.listdir(maildir_path):
        user_dir = os.path.join(maildir_path, user)
        if not os.path.isdir(user_dir):
            continue
            
        # Process sent emails to identify the sender
        sent_dir = os.path.join(user_dir, "sent")
        if os.path.exists(sent_dir) and os.path.isdir(sent_dir):
            for file_name in os.listdir(sent_dir):
                file_path = os.path.join(sent_dir, file_name)
                if os.path.isfile(file_path):
                    content = extract_email_content(file_path)
                    if content:
                        emails_data.append({
                            'sender': user,
                            'content': content,
                            'path': file_path
                        })
        
        # Also check sent_items directory
        sent_items_dir = os.path.join(user_dir, "sent_items")
        if os.path.exists(sent_items_dir) and os.path.isdir(sent_items_dir):
            for file_name in os.listdir(sent_items_dir):
                file_path = os.path.join(sent_items_dir, file_name)
                if os.path.isfile(file_path):
                    content = extract_email_content(file_path)
                    if content:
                        emails_data.append({
                            'sender': user,
                            'content': content,
                            'path': file_path
                        })
    
    # Create DataFrame and save processed data
    df = pd.DataFrame(emails_data)
    
    # Filter to include only top NUM_TOP_USERS users by email count
    top_users = df['sender'].value_counts().head(NUM_TOP_USERS).index.tolist()
    df = df[df['sender'].isin(top_users)]
    
    # Save processed data
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_FILE}")
    print(f"Dataset contains {len(df)} emails from {len(df['sender'].unique())} unique senders")
    
    return df

def create_and_save_train_test_split(emails_df, test_size=0.2, random_state=42):
    """Create and save train/test splits."""
    print("Creating and saving train/test splits...")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        emails_df['content'], 
        emails_df['sender'],
        test_size=test_size, 
        random_state=random_state,
        stratify=emails_df['sender']
    )
    
    # Create DataFrames for train and test sets
    train_df = pd.DataFrame({
        'content': X_train,
        'sender': y_train
    })
    
    test_df = pd.DataFrame({
        'content': X_test,
        'sender': y_test
    })
    
    # Save the splits
    train_df.to_csv(TRAIN_DATA_FILE, index=False)
    test_df.to_csv(TEST_DATA_FILE, index=False)
    
    print(f"Train data saved to {TRAIN_DATA_FILE} ({len(train_df)} samples)")
    print(f"Test data saved to {TEST_DATA_FILE} ({len(test_df)} samples)")
    
    return train_df, test_df

def load_train_test_split():
    """Load existing train/test splits."""
    if not os.path.exists(TRAIN_DATA_FILE) or not os.path.exists(TEST_DATA_FILE):
        return None, None
    
    print(f"Loading train/test splits from saved files...")
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)
    
    print(f"Loaded train data: {len(train_df)} samples")
    print(f"Loaded test data: {len(test_df)} samples")
    
    return train_df, test_df

def get_or_create_train_test_split(emails_df):
    """Get existing train/test split or create new one if it doesn't exist."""
    train_df, test_df = load_train_test_split()
    
    if train_df is None or test_df is None:
        print("No existing train/test split found. Creating new split...")
        train_df, test_df = create_and_save_train_test_split(emails_df)
    else:
        print("Using existing train/test split.")
        # Verify that the split is still valid for the current dataset
        expected_total = len(emails_df)
        actual_total = len(train_df) + len(test_df)
        
        if abs(expected_total - actual_total) > 100:  # Allow some tolerance
            print(f"Warning: Train/test split size ({actual_total}) doesn't match current dataset ({expected_total})")
            print("Creating new train/test split...")
            train_df, test_df = create_and_save_train_test_split(emails_df)
    
    return train_df, test_df

def build_naive_bayes_classifier(train_df, test_df, save=True):
    """Build and evaluate a Naive Bayes classifier."""
    # Create a pipeline with TF-IDF and Multinomial Naive Bayes
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )),
        ('classifier', MultinomialNB())
    ])
    
    # Train the model
    print("Training the Naive Bayes classifier...")
    start_time = time.time()
    pipeline.fit(train_df['content'], train_df['sender'])
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    y_pred = pipeline.predict(test_df['content'])
    accuracy = accuracy_score(test_df['sender'], y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(test_df['sender'], y_pred))
    
    # Save the model if requested
    if save:
        # Add training time to metadata
        metadata_extras = {'training_time': training_time}
        save_model(pipeline, BASELINE_MODEL_FILE, "baseline", accuracy, train_df, metadata_extras)
    
    return pipeline

def optimize_hyperparameters(train_df, test_df, save=True):
    """Optimize hyperparameters using GridSearchCV."""
    # Use the pre-split data
    X_train, y_train = train_df['content'], train_df['sender']
    X_test, y_test = test_df['content'], test_df['sender']
    
    # Define pipeline
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', MultinomialNB())
    ])
    
    # Define parameter grid
    param_grid = {
        'vectorizer__max_features': [3000, 5000],
        'vectorizer__min_df': [2, 3],
        'vectorizer__max_df': [0.7, 0.8],
        'classifier__alpha': [0.01, 0.1, 1.0],
    }
    
    # Perform grid search
    print("Optimizing hyperparameters...")
    start_time = time.time()
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Print best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Optimization completed in {training_time:.2f} seconds")
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy with optimized model: {accuracy:.4f}")
    
    # Save the model if requested
    if save:
        # Add training time and best parameters to metadata
        metadata_extras = {
            'training_time': training_time,
            'best_parameters': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_
        }
        save_model(best_model, OPTIMIZED_MODEL_FILE, "optimized", accuracy, train_df, metadata_extras)
    
    return best_model

def predict_sender(model, email_content):
    """Predict the sender of a given email."""
    # Clean the email content
    cleaned_content = re.sub(r'\s+', ' ', email_content)
    cleaned_content = re.sub(r'[^\w\s]', '', cleaned_content)
    cleaned_content = cleaned_content.lower()
    
    # Make prediction
    sender = model.predict([cleaned_content])[0]
    probabilities = model.predict_proba([cleaned_content])[0]
    confidence = max(probabilities)
    
    return sender, confidence

def evaluate_model_comprehensive(model, test_df, model_name="Model"):
    """Comprehensive evaluation of a model with detailed metrics."""
    print(f"\n=== Comprehensive Evaluation: {model_name} ===")
    
    # Make predictions
    y_true = test_df['sender']
    y_pred = model.predict(test_df['content'])
    y_proba = model.predict_proba(test_df['content'])
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_weighted, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Create results dictionary
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1_weighted,
        'f1_macro': macro_f1,
        'num_classes': len(np.unique(y_true)),
        'num_samples': len(y_true),
        'evaluation_time': time.time()
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1_weighted:.4f}")
    print(f"F1-score (macro): {macro_f1:.4f}")
    print(f"Number of classes: {len(np.unique(y_true))}")
    
    # Detailed classification report
    class_report = classification_report(y_true, y_pred, output_dict=True)
    results['classification_report'] = class_report
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results['confusion_matrix'] = cm.tolist()  # Convert to list for JSON serialization
    
    # Top performing classes
    class_f1_scores = []
    for class_name, metrics in class_report.items():
        if isinstance(metrics, dict) and 'f1-score' in metrics:
            class_f1_scores.append((class_name, metrics['f1-score'], metrics['support']))
    
    class_f1_scores.sort(key=lambda x: x[1], reverse=True)
    results['top_classes'] = class_f1_scores[:10]
    results['worst_classes'] = class_f1_scores[-10:]
    
    print(f"\nTop 5 performing classes (by F1-score):")
    for class_name, f1_val, support in class_f1_scores[:5]:
        print(f"  {class_name}: F1={f1_val:.3f} (support={support})")
    
    print(f"\nWorst 5 performing classes (by F1-score):")
    for class_name, f1_val, support in class_f1_scores[-5:]:
        print(f"  {class_name}: F1={f1_val:.3f} (support={support})")
    
    return results, y_true, y_pred, y_proba

def create_confusion_matrix_plot(y_true, y_pred, model_name, save_path=None, top_n=20):
    """Create and save confusion matrix visualization."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Skipping confusion matrix plot.")
        return None
        
    # Get top N most frequent classes for readability
    top_classes = pd.Series(y_true).value_counts().head(top_n).index.tolist()
    
    # Filter data to top classes
    mask = pd.Series(y_true).isin(top_classes) & pd.Series(y_pred).isin(top_classes)
    y_true_filtered = pd.Series(y_true)[mask]
    y_pred_filtered = pd.Series(y_pred)[mask]
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_classes)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=top_classes, yticklabels=top_classes)
    plt.title(f'Confusion Matrix - {model_name}\n(Top {top_n} Most Frequent Classes)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    return plt

def create_performance_comparison_plot(baseline_results, optimized_results, save_path=None):
    """Create a comparison plot of model performances."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Skipping performance comparison plot.")
        return None
        
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'f1_macro']
    baseline_values = [baseline_results[metric] for metric in metrics]
    optimized_values = [optimized_results[metric] for metric in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline Model', alpha=0.8)
    bars2 = ax.bar(x + width/2, optimized_values, width, label='Optimized Model', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison saved to {save_path}")
    
    return plt

def create_class_performance_plot(results, save_path=None, top_n=15):
    """Create a plot showing per-class F1 scores."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Skipping class performance plot.")
        return None
        
    class_report = results['classification_report']
    
    # Extract class performance data
    classes = []
    f1_scores = []
    supports = []
    
    for class_name, metrics in class_report.items():
        if isinstance(metrics, dict) and 'f1-score' in metrics:
            classes.append(class_name)
            f1_scores.append(metrics['f1-score'])
            supports.append(metrics['support'])
    
    # Sort by F1 score and take top N
    class_data = list(zip(classes, f1_scores, supports))
    class_data.sort(key=lambda x: x[1], reverse=True)
    
    top_classes = class_data[:top_n]
    worst_classes = class_data[-top_n:]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top performing classes
    top_names, top_f1s, top_supports = zip(*top_classes)
    bars1 = ax1.barh(range(len(top_names)), top_f1s, alpha=0.8, color='green')
    ax1.set_yticks(range(len(top_names)))
    ax1.set_yticklabels(top_names)
    ax1.set_xlabel('F1 Score')
    ax1.set_title(f'Top {top_n} Performing Classes')
    ax1.set_xlim(0, 1)
    
    # Add support information
    for i, (f1, support) in enumerate(zip(top_f1s, top_supports)):
        ax1.text(f1 + 0.01, i, f'(n={support})', va='center', fontsize=8)
    
    # Worst performing classes
    worst_names, worst_f1s, worst_supports = zip(*worst_classes)
    bars2 = ax2.barh(range(len(worst_names)), worst_f1s, alpha=0.8, color='red')
    ax2.set_yticks(range(len(worst_names)))
    ax2.set_yticklabels(worst_names)
    ax2.set_xlabel('F1 Score')
    ax2.set_title(f'Worst {top_n} Performing Classes')
    ax2.set_xlim(0, 1)
    
    # Add support information
    for i, (f1, support) in enumerate(zip(worst_f1s, worst_supports)):
        ax2.text(f1 + 0.01, i, f'(n={support})', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class performance plot saved to {save_path}")
    
    return plt

def export_results_to_files(baseline_results, optimized_results, train_df, test_df):
    """Export all results to various file formats."""
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Export detailed metrics to JSON
    combined_results = {
        'evaluation_timestamp': timestamp,
        'dataset_info': {
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'num_classes': len(train_df['sender'].unique()),
            'top_senders': train_df['sender'].value_counts().head(10).to_dict()
        },
        'baseline_model': baseline_results,
        'optimized_model': optimized_results
    }
    
    json_path = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)
    print(f"Detailed results saved to {json_path}")
    
    # 2. Export summary metrics to CSV
    summary_data = []
    for model_name, results in [('Baseline', baseline_results), ('Optimized', optimized_results)]:
        summary_data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision (Weighted)': results['precision_weighted'],
            'Recall (Weighted)': results['recall_weighted'],
            'F1-Score (Weighted)': results['f1_weighted'],
            'F1-Score (Macro)': results['f1_macro'],
            'Number of Classes': results['num_classes'],
            'Test Samples': results['num_samples']
        })
    
    summary_df = pd.DataFrame(summary_data)
    csv_path = os.path.join(RESULTS_DIR, f"model_comparison_{timestamp}.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Model comparison saved to {csv_path}")
    
    # 3. Export detailed classification reports
    for model_name, results in [('baseline', baseline_results), ('optimized', optimized_results)]:
        class_report_df = pd.DataFrame(results['classification_report']).T
        class_report_path = os.path.join(RESULTS_DIR, f"{model_name}_classification_report_{timestamp}.csv")
        class_report_df.to_csv(class_report_path)
        print(f"{model_name.title()} classification report saved to {class_report_path}")
    
    return json_path, csv_path

def generate_comprehensive_report():
    """Generate a comprehensive evaluation report with visualizations."""
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE EVALUATION REPORT")
    print("="*60)
    
    # Check if we have trained models
    if not (os.path.exists(BASELINE_MODEL_FILE) and os.path.exists(OPTIMIZED_MODEL_FILE)):
        print("Error: No trained models found. Please train models first.")
        return None
    
    # Check if we have train/test splits
    if not (os.path.exists(TRAIN_DATA_FILE) and os.path.exists(TEST_DATA_FILE)):
        print("Error: No train/test split found. Please process data first.")
        return None
    
    # Load models and data
    print("Loading models and data...")
    baseline_model = load_model(BASELINE_MODEL_FILE)
    optimized_model = load_model(OPTIMIZED_MODEL_FILE)
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)
    
    # Evaluate models
    baseline_results, y_true_base, y_pred_base, y_proba_base = evaluate_model_comprehensive(
        baseline_model, test_df, "Baseline Model")
    
    optimized_results, y_true_opt, y_pred_opt, y_proba_opt = evaluate_model_comprehensive(
        optimized_model, test_df, "Optimized Model")
    
    # Create visualizations
    print("\nCreating visualizations...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plots_generated = []
    
    if VISUALIZATION_AVAILABLE:
        # 1. Confusion matrices
        cm_baseline_path = os.path.join(RESULTS_DIR, f"confusion_matrix_baseline_{timestamp}.png")
        plt1 = create_confusion_matrix_plot(y_true_base, y_pred_base, "Baseline Model", cm_baseline_path)
        if plt1: 
            plt1.close()
            plots_generated.append(cm_baseline_path)
        
        cm_optimized_path = os.path.join(RESULTS_DIR, f"confusion_matrix_optimized_{timestamp}.png")
        plt2 = create_confusion_matrix_plot(y_true_opt, y_pred_opt, "Optimized Model", cm_optimized_path)
        if plt2:
            plt2.close()
            plots_generated.append(cm_optimized_path)
        
        # 2. Performance comparison
        comparison_path = os.path.join(RESULTS_DIR, f"performance_comparison_{timestamp}.png")
        plt3 = create_performance_comparison_plot(baseline_results, optimized_results, comparison_path)
        if plt3:
            plt3.close()
            plots_generated.append(comparison_path)
        
        # 3. Class performance plots
        class_perf_baseline_path = os.path.join(RESULTS_DIR, f"class_performance_baseline_{timestamp}.png")
        plt4 = create_class_performance_plot(baseline_results, class_perf_baseline_path)
        if plt4:
            plt4.close()
            plots_generated.append(class_perf_baseline_path)
        
        class_perf_optimized_path = os.path.join(RESULTS_DIR, f"class_performance_optimized_{timestamp}.png")
        plt5 = create_class_performance_plot(optimized_results, class_perf_optimized_path)
        if plt5:
            plt5.close()
            plots_generated.append(class_perf_optimized_path)
    else:
        print("Visualization libraries not available. Skipping plot generation.")
        print("To enable visualizations, install: pip install matplotlib seaborn")
    
    # Export results to files
    print("\nExporting results to files...")
    json_path, csv_path = export_results_to_files(baseline_results, optimized_results, train_df, test_df)
    
    # Generate summary report
    print("\n" + "="*60)
    print("EVALUATION SUMMARY REPORT")
    print("="*60)
    print(f"Evaluation completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {len(train_df)} training samples, {len(test_df)} testing samples")
    print(f"Number of classes (senders): {len(train_df['sender'].unique())}")
    
    print(f"\nBASELINE MODEL PERFORMANCE:")
    print(f"  Accuracy: {baseline_results['accuracy']:.4f}")
    print(f"  Precision (weighted): {baseline_results['precision_weighted']:.4f}")
    print(f"  Recall (weighted): {baseline_results['recall_weighted']:.4f}")
    print(f"  F1-score (weighted): {baseline_results['f1_weighted']:.4f}")
    print(f"  F1-score (macro): {baseline_results['f1_macro']:.4f}")
    
    print(f"\nOPTIMIZED MODEL PERFORMANCE:")
    print(f"  Accuracy: {optimized_results['accuracy']:.4f}")
    print(f"  Precision (weighted): {optimized_results['precision_weighted']:.4f}")
    print(f"  Recall (weighted): {optimized_results['recall_weighted']:.4f}")
    print(f"  F1-score (weighted): {optimized_results['f1_weighted']:.4f}")
    print(f"  F1-score (macro): {optimized_results['f1_macro']:.4f}")
    
    # Improvement analysis
    accuracy_improvement = optimized_results['accuracy'] - baseline_results['accuracy']
    f1_improvement = optimized_results['f1_weighted'] - baseline_results['f1_weighted']
    
    print(f"\nIMPROVEMENT ANALYSIS:")
    print(f"  Accuracy improvement: {accuracy_improvement:+.4f} ({accuracy_improvement/baseline_results['accuracy']*100:+.2f}%)")
    print(f"  F1-score improvement: {f1_improvement:+.4f} ({f1_improvement/baseline_results['f1_weighted']*100:+.2f}%)")
    
    print(f"\nFILES GENERATED:")
    print(f"  Detailed results (JSON): {json_path}")
    print(f"  Model comparison (CSV): {csv_path}")
    
    if plots_generated:
        for plot_path in plots_generated:
            plot_name = os.path.basename(plot_path).replace(f"_{timestamp}.png", "").replace("_", " ").title()
            print(f"  {plot_name}: {plot_path}")
    else:
        print("  No plots generated (visualization libraries not available)")
    
    print(f"\nAll results saved in: {RESULTS_DIR}/")
    print("="*60)
    
    return {
        'baseline_results': baseline_results,
        'optimized_results': optimized_results,
        'files_generated': {
            'json': json_path,
            'csv': csv_path,
            'plots': plots_generated
        }
    }

def predict_sender(model, email_content):
    """Predict the sender of a given email."""
    # Clean the email content
    cleaned_content = re.sub(r'\s+', ' ', email_content)
    cleaned_content = re.sub(r'[^\w\s]', '', cleaned_content)
    cleaned_content = cleaned_content.lower()
    
    # Make prediction
    sender = model.predict([cleaned_content])[0]
    probabilities = model.predict_proba([cleaned_content])[0]
    confidence = max(probabilities)
    
    return sender, confidence

def save_model(model, filepath, model_type, accuracy, emails_df, metadata_extras=None):
    """Save a trained model to disk."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the model
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")
    
    # Save metadata
    metadata = {
        "model_type": model_type,
        "num_users": len(emails_df['sender'].unique()),
        "num_emails": len(emails_df),
        "accuracy": accuracy,
        "date_trained": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "top_users": emails_df['sender'].value_counts().head(10).to_dict()
    }
    
    # Add extra metadata if provided
    if metadata_extras:
        metadata.update(metadata_extras)
    
    # If first model being saved, create new metadata file
    if os.path.exists(MODEL_METADATA_FILE):
        try:
            with open(MODEL_METADATA_FILE, 'r') as f:
                existing_metadata = pd.read_json(f, orient='records').to_dict('records')[0]
            existing_metadata[model_type] = metadata
            pd.DataFrame([existing_metadata]).to_json(MODEL_METADATA_FILE, orient='records')
        except:
            pd.DataFrame([{model_type: metadata}]).to_json(MODEL_METADATA_FILE, orient='records')
    else:
        pd.DataFrame([{model_type: metadata}]).to_json(MODEL_METADATA_FILE, orient='records')
    
    print(f"Model metadata saved to {MODEL_METADATA_FILE}")

def load_model(filepath):
    """Load a trained model from disk."""
    if not os.path.exists(filepath):
        return None
    
    print(f"Loading model from {filepath}")
    return joblib.load(filepath)

def get_model_metadata():
    """Get metadata about saved models."""
    if not os.path.exists(MODEL_METADATA_FILE):
        return None
    
    try:
        with open(MODEL_METADATA_FILE, 'r') as f:
            return pd.read_json(f, orient='records').to_dict('records')[0]
    except:
        return None

def show_model_info():
    """Display information about the available trained models."""
    metadata = get_model_metadata()
    if not metadata:
        print("No trained models found.")
        return
    
    print("\n=== Trained Model Information ===")
    for model_type, info in metadata.items():
        print(f"\nModel Type: {model_type.upper()}")
        print(f"  Trained on: {info['date_trained']}")
        print(f"  Number of users: {info['num_users']}")
        print(f"  Number of emails: {info['num_emails']}")
        print(f"  Accuracy: {info['accuracy']:.4f}")
        print("  Top users (number of emails):")
        for user, count in list(info['top_users'].items())[:5]:
            print(f"    - {user}: {count}")
    print()

print()

def show_split_info():
    """Display information about the train/test split."""
    if not os.path.exists(TRAIN_DATA_FILE) or not os.path.exists(TEST_DATA_FILE):
        print("No train/test split files found.")
        return
    
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)
    
    print("\n=== Train/Test Split Information ===")
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Total: {len(train_df) + len(test_df)} samples")
    print(f"Test ratio: {len(test_df) / (len(train_df) + len(test_df)):.1%}")
    print(f"Unique senders: {len(train_df['sender'].unique())}")
    
    # Show distribution of top senders
    print("\nTop senders in training set:")
    for sender, count in train_df['sender'].value_counts().head(5).items():
        test_count = test_df[test_df['sender'] == sender].shape[0]
        print(f"  - {sender}: {count} train, {test_count} test")

def train_models(emails_df):
    """Train both baseline and optimized models."""
    # Get or create train/test split
    train_df, test_df = get_or_create_train_test_split(emails_df)
    
    # Train baseline model
    print("\n=== Training Baseline Model ===")
    baseline_model = build_naive_bayes_classifier(train_df, test_df)
    
    # Optimize hyperparameters (optional, can be time-consuming)
    print("\n=== Optimizing Hyperparameters ===")
    optimized_model = optimize_hyperparameters(train_df, test_df)
    
    return baseline_model, optimized_model

def main():
    # Check if running in a virtual environment
    if not os.environ.get('VIRTUAL_ENV'):
        print("Warning: It's recommended to run this script in a virtual environment.")
        print("See the setup instructions at the top of this file.")
        print("- On Windows: run setup.bat and then run.bat")
        print("- On macOS/Linux: run ./setup.sh and then ./run.sh")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please set up and activate a virtual environment before running again.")
            return
    
    # Use existing maildir path instead of downloading
    maildir_path = os.path.join(os.getcwd(), "maildir")
    if not os.path.exists(maildir_path):
        print(f"Error: Could not find the maildir folder at {maildir_path}")
        print("Please make sure the Enron email dataset 'maildir' folder exists in the current directory.")
        return
    
    # Check if models exist
    has_saved_models = os.path.exists(BASELINE_MODEL_FILE) and os.path.exists(OPTIMIZED_MODEL_FILE)
    has_saved_splits = os.path.exists(TRAIN_DATA_FILE) and os.path.exists(TEST_DATA_FILE)
    
    if has_saved_models:
        # Display saved model information
        show_model_info()
        
        # Display train/test split information
        if has_saved_splits:
            show_split_info()
        
        # Ask if user wants to use saved models
        print("\nFound previously trained models.")
        if has_saved_splits:
            choice = input("Would you like to (1) use saved models, (2) retrain models with existing split, (3) recreate split and retrain, or (4) generate evaluation report? Enter 1, 2, 3, or 4: ")
        else:
            choice = input("Would you like to (1) use saved models, (2) retrain models, or (3) generate evaluation report? Enter 1, 2, or 3: ")
        
        if choice == '4' or (choice == '3' and not has_saved_splits):
            # Generate comprehensive evaluation report
            report_results = generate_comprehensive_report()
            if report_results:
                print("\nEvaluation report generated successfully!")
                response = input("Would you like to continue with interactive testing? (y/n): ")
                if response.lower() != 'y':
                    return
                
                # Load data and models for interactive testing
                emails_df = process_emails(maildir_path)
                optimized_model = load_model(OPTIMIZED_MODEL_FILE)
            else:
                return
        elif choice == '1':
            # Load emails dataset for sample prediction
            emails_df = process_emails(maildir_path)
            
            # Load saved models
            print("\n=== Loading Saved Models ===")
            baseline_model = load_model(BASELINE_MODEL_FILE)
            optimized_model = load_model(OPTIMIZED_MODEL_FILE)
            
        elif choice == '1':
            # Load emails dataset for sample prediction
            emails_df = process_emails(maildir_path)
            
            # Load saved models
            print("\n=== Loading Saved Models ===")
            baseline_model = load_model(BASELINE_MODEL_FILE)
            optimized_model = load_model(OPTIMIZED_MODEL_FILE)
            
            if baseline_model is None or optimized_model is None:
                print("Error loading models. Will train new models.")
                baseline_model, optimized_model = train_models(emails_df)
        elif choice == '3' and has_saved_splits:
            # Recreate train/test split and retrain models
            print(f"Using existing maildir data at: {maildir_path}")
            emails_df = process_emails(maildir_path)
            
            # Remove existing split files to force recreation
            if os.path.exists(TRAIN_DATA_FILE):
                os.remove(TRAIN_DATA_FILE)
            if os.path.exists(TEST_DATA_FILE):
                os.remove(TEST_DATA_FILE)
            print("Removed existing train/test split files.")
            
            baseline_model, optimized_model = train_models(emails_df)
        else:
            # Process emails and train models (choice == '2' or choice == '2' when no splits)
            print(f"Using existing maildir data at: {maildir_path}")
            emails_df = process_emails(maildir_path)
            baseline_model, optimized_model = train_models(emails_df)
    else:
        # No saved models found, process emails and train models
        print(f"Using existing maildir data at: {maildir_path}")
        emails_df = process_emails(maildir_path)
        baseline_model, optimized_model = train_models(emails_df)
    
    # Example prediction
    print("\n=== Example Prediction ===")
    sample_email = emails_df.iloc[0]['content']
    sender, confidence = predict_sender(optimized_model, sample_email)
    print(f"Sample email snippet: {sample_email[:100]}...")
    print(f"Predicted sender: {sender} (confidence: {confidence:.4f})")
    
    # Allow for user input to test the model
    print("\n=== Interactive Testing ===")
    while True:
        user_input = input("\nEnter an email text to predict the sender (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        sender, confidence = predict_sender(optimized_model, user_input)
        print(f"Predicted sender: {sender} (confidence: {confidence:.4f})")

if __name__ == "__main__":
    main()