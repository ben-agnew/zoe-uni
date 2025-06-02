#!/usr/bin/env python3
"""
Simplified Naive Bayes Email Classifier
A streamlined script for creating datasets, training, and evaluating a Naive Bayes classifier
on the Enron email dataset for sender identification.

This script focuses on:
1. Dataset creation and preprocessing
2. Train/validation/test split
3. Naive Bayes training (baseline and optimized)
4. Model evaluation and results

Usage:
    python simple_naive_bayes.py

Requirements:
    - Python 3.8+
    - scikit-learn
    - pandas
    - numpy
    - maildir folder with Enron dataset
"""

import email
import json
import os
import re
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Configuration constants
PROCESSED_DATA_FILE = "processed_emails.csv"
TRAIN_DATA_FILE = "train_data.csv"
VALIDATION_DATA_FILE = "validation_data.csv"
TEST_DATA_FILE = "test_data.csv"
BASELINE_MODEL_FILE = "baseline_model.joblib"
OPTIMIZED_MODEL_FILE = "optimized_model.joblib"
RESULTS_FILE = "results.json"

# Dataset parameters
NUM_TOP_USERS = 150  # Number of top email senders to include
MIN_EMAILS_PER_USER = 20  # Minimum emails required per user


def extract_email_content(email_path):
    """Extract clean text content from an email file, removing forwarded content."""
    try:
        with open(email_path, 'r', encoding='utf-8', errors='ignore') as file:
            # Parse the email
            msg = email.message_from_file(file)
            
            # Extract the body
            if msg.is_multipart():
                content = ""
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        content += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            else:
                content = msg.get_payload(decode=True)
                if content:
                    content = content.decode('utf-8', errors='ignore')
                else:
                    content = str(msg.get_payload())
            
            if not content:
                return None
            
            # Clean the content line by line and stop at forwarded content
            lines = content.split('\n')
            cleaned_lines = []
            header_section = True
            
            for line in lines:
                line = line.strip()
                
                # Check if this line indicates forwarded content - if so, stop processing
                if _is_forwarded_marker(line):
                    break
                
                # Skip empty lines in header section
                if header_section and not line:
                    continue
                
                # Enhanced header detection - check for email header patterns
                is_header_line = _is_email_header_line(line)
                
                # Detect end of header section
                if header_section and line and not is_header_line:
                    header_section = False
                
                # Skip header lines (both in header section and scattered throughout)
                if is_header_line:
                    continue
                
                # Skip quoted text (replies)
                if line.startswith('>') or line.startswith('|'):
                    continue
                
                # Skip lines that are mostly punctuation or special characters
                if len(line) > 0 and len(re.sub(r'[^a-zA-Z0-9\s]', '', line)) / len(line) < 0.5:
                    continue
                
                # Clean the line
                line = re.sub(r'\s+', ' ', line)
                
                if line and len(line) > 2:  # Skip very short lines
                    cleaned_lines.append(line)
            
            # Join lines and do final cleaning
            content = ' '.join(cleaned_lines)
            
            # Remove URLs
            content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
            
            # Remove email addresses
            content = re.sub(r'\S+@\S+', '', content)
            
            # Remove phone numbers
            content = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', content)
            
            # Remove remaining header-like patterns that might have slipped through
            # Pattern: "Word:" or "Word Word:" at the beginning of sentences
            content = re.sub(r'\b[A-Za-z]{2,15}(?:\s+[A-Za-z]{2,15})?:\s*', '', content)
            
            # Remove patterns like "To: Name cc:" or "From: Name"
            content = re.sub(r'\b(?:To|From|Cc|Bcc|Subject|Date|Sent|Received):\s*[^.!?]*?(?=\s|$)', '', content, flags=re.IGNORECASE)
            
            # Remove excess whitespace
            content = re.sub(r'\s+', ' ', content).strip()
            
            return content if len(content) > 50 else None  # Filter very short emails
            
    except Exception as e:
        print(f"Error processing {email_path}: {str(e)}")
        return None


def _is_forwarded_marker(line):
    """Check if a line indicates the start of forwarded content."""
    line_lower = line.lower().strip()
    
    # Simple and fast checks for common forwarded email markers
    forwarded_indicators = [
        '-----original message-----',
        'begin forwarded message',
        'forwarded message',
        '-----forwarded by',
        'original message',
        'forwarded by'
    ]
    
    # Check if line starts with or contains these indicators
    for indicator in forwarded_indicators:
        if indicator in line_lower:
            return True
    
    # Check for "From: ... Sent: ... To: ..." pattern (all on same line)
    if line_lower.startswith('from:') and 'sent:' in line_lower and 'to:' in line_lower:
        return True
    
    # Check for "On ... wrote:" pattern
    if line_lower.startswith('on ') and 'wrote:' in line_lower:
        return True
    
    return False


def _is_email_header_line(line):
    """Check if a line looks like an email header that should be filtered out."""
    if not line:
        return False
    
    line_lower = line.lower().strip()
    
    # Common email headers (case insensitive)
    header_prefixes = [
        'from:', 'to:', 'cc:', 'bcc:', 'subject:', 'date:', 'sent:', 'received:',
        'reply-to:', 'return-path:', 'message-id:', 'in-reply-to:', 'references:',
        'mime-version:', 'content-type:', 'content-transfer-encoding:',
        'x-originating-ip:', 'x-mailer:', 'importance:', 'priority:',
        'envelope-to:', 'delivery-date:', 'received-spf:'
    ]
    
    # Check if line starts with any email header
    for prefix in header_prefixes:
        if line_lower.startswith(prefix):
            return True
    
    # Check for standalone "cc:" or "bcc:" that might appear on their own line
    if line_lower in ['cc:', 'bcc:', 'to:', 'from:']:
        return True
    
    # Check for header-like patterns (word followed by colon and content)
    # but be careful not to catch legitimate content
    if ':' in line and len(line) < 100:  # Limit to reasonably short lines
        parts = line.split(':', 1)
        if len(parts) == 2:
            header_part = parts[0].strip().lower()
            # Check if it looks like a header (single word or two words max)
            if (len(header_part.split()) <= 2 and 
                not any(char.isdigit() for char in header_part) and  # No numbers
                len(header_part) < 20):  # Reasonable header length
                # Additional check: if it contains names that look like recipients
                content_part = parts[1].strip()
                if (any(word in content_part.lower() for word in ['@', 'enron.com']) or
                    re.match(r'^[A-Za-z\s,.-]+$', content_part[:50])):  # Looks like names/emails
                    return True
    
    return False


def _check_if_forwarded(email_path):
    """Check if an email file contains forwarded content with efficient partial reading."""
    try:
        with open(email_path, 'r', encoding='utf-8', errors='ignore') as file:
            # Read only first 1500 characters to check for forwarded indicators
            content = file.read(1500).lower()
            
            # Quick string-based checks (no regex for speed)
            forwarded_indicators = [
                '-----original message-----',
                'begin forwarded message',
                'forwarded message',
                '-----forwarded by',
                'original message',
                'forwarded by'
            ]
            
            for indicator in forwarded_indicators:
                if indicator in content:
                    return True
            
            # Simple checks without regex
            if 'from:' in content and 'sent:' in content and 'to:' in content:
                return True
            
            if 'wrote:' in content and ('on ' in content[:500]):  # Check 'on' in first 500 chars only
                return True
                
            return False
            
    except Exception:
        return False


def create_dataset(maildir_path, force_recreate=False):
    """Create dataset from maildir emails."""
    print(f"Creating dataset from maildir at: {maildir_path}")
    
    if os.path.exists(PROCESSED_DATA_FILE) and not force_recreate:
        print(f"Loading existing processed data from {PROCESSED_DATA_FILE}")
        return pd.read_csv(PROCESSED_DATA_FILE)
    
    emails_data = []
    total_users = 0
    processed_users = 0
    total_email_files = 0
    rejected_emails = 0
    forwarded_emails_removed = 0
    
    # Process each user's emails
    for user in os.listdir(maildir_path):
        user_dir = os.path.join(maildir_path, user)
        if not os.path.isdir(user_dir):
            continue
        
        total_users += 1
        print(f"Processing user {total_users}: {user}")
        user_emails = []
        user_total_files = 0
        user_rejected = 0
        user_forwarded = 0
        
        # Check for sent email directories
        sent_directories = ["sent", "sent_items", "_sent_mail"]
        
        for sent_dir_name in sent_directories:
            sent_dir = os.path.join(user_dir, sent_dir_name)
            if os.path.exists(sent_dir) and os.path.isdir(sent_dir):
                print(f"  Found {sent_dir_name} directory")
                
                # Process each email file
                for file_name in os.listdir(sent_dir):
                    file_path = os.path.join(sent_dir, file_name)
                    if os.path.isfile(file_path):
                        user_total_files += 1
                        total_email_files += 1
                        
                        # Check if email contains forwarded content before processing
                        contains_forwarded = _check_if_forwarded(file_path)
                        if contains_forwarded:
                            forwarded_emails_removed += 1
                            user_forwarded += 1
                        
                        content = extract_email_content(file_path)
                        if content and content.strip():
                            user_emails.append({
                                'sender': user,
                                'content': content,
                                'file_path': file_path
                            })
                        else:
                            user_rejected += 1
        
        # Track total rejected emails
        rejected_emails += user_rejected
        print(f"  Processed {user_total_files} files, accepted {len(user_emails)}, forwarded {user_forwarded}, rejected {user_rejected}")
        
        # Only include users with sufficient emails
        if len(user_emails) >= MIN_EMAILS_PER_USER:
            print(f"  ✓ Added {len(user_emails)} emails for {user}")
            emails_data.extend(user_emails)
            processed_users += 1
        else:
            print(f"  ✗ Skipped {user} (only {len(user_emails)} emails, minimum: {MIN_EMAILS_PER_USER})")
    
    print(f"\nProcessed {total_users} users, included {processed_users} users")
    print(f"Email processing statistics:")
    print(f"  Total email files processed: {total_email_files:,}")
    print(f"  Emails with forwarded content: {forwarded_emails_removed:,} ({forwarded_emails_removed/total_email_files*100:.1f}%)")
    print(f"  Emails rejected (too short/invalid): {rejected_emails:,} ({rejected_emails/total_email_files*100:.1f}%)")
    print(f"  Emails accepted: {total_email_files - rejected_emails:,} ({(total_email_files - rejected_emails)/total_email_files*100:.1f}%)")
    
    # Create DataFrame
    df = pd.DataFrame(emails_data)
    print(f"Initial dataset: {len(df)} emails from {len(df['sender'].unique())} users")
    
    # Remove duplicate emails
    print("Removing duplicate emails...")
    initial_count = len(df)
    df = df.drop_duplicates(subset=['content'], keep='first')
    duplicates_removed = initial_count - len(df)
    print(f"Removed {duplicates_removed} duplicate emails")
    
    # Verify users still meet minimum requirement after duplicate removal
    user_counts = df['sender'].value_counts()
    users_to_remove = user_counts[user_counts < MIN_EMAILS_PER_USER].index.tolist()
    
    if users_to_remove:
        print(f"Removing {len(users_to_remove)} users who now have < {MIN_EMAILS_PER_USER} emails")
        df = df[~df['sender'].isin(users_to_remove)]
    
    # Keep only top NUM_TOP_USERS by email count
    final_user_counts = df['sender'].value_counts()
    top_users = final_user_counts.head(NUM_TOP_USERS).index.tolist()
    df = df[df['sender'].isin(top_users)]
    
    # Remove file_path column (not needed for modeling)
    df = df.drop('file_path', axis=1)
    
    # Save processed dataset
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"\n✓ Dataset created and saved to {PROCESSED_DATA_FILE}")
    print(f"Final dataset: {len(df)} emails from {len(df['sender'].unique())} unique senders")
    
    # Show dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Total emails: {len(df):,}")
    print(f"  Unique senders: {len(df['sender'].unique())}")
    print(f"  Min emails per sender: {final_user_counts.min()}")
    print(f"  Max emails per sender: {final_user_counts.max()}")
    print(f"  Average emails per sender: {final_user_counts.mean():.1f}")
    print(f"  Median emails per sender: {final_user_counts.median():.1f}")
    
    return df


def create_train_test_split(emails_df, test_size=0.2, validation_size=0.15, random_state=42, force_recreate=False):
    """Create stratified train/validation/test splits."""
    print(f"\nCreating train/validation/test splits...")
    
    # Check if splits already exist
    if (os.path.exists(TRAIN_DATA_FILE) and 
        os.path.exists(VALIDATION_DATA_FILE) and 
        os.path.exists(TEST_DATA_FILE) and not force_recreate):
        print("Loading existing splits...")
        train_df = pd.read_csv(TRAIN_DATA_FILE)
        validation_df = pd.read_csv(VALIDATION_DATA_FILE)
        test_df = pd.read_csv(TEST_DATA_FILE)
        return train_df, validation_df, test_df
    
    # Create new splits
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        emails_df['content'],
        emails_df['sender'],
        test_size=test_size,
        random_state=random_state,
        stratify=emails_df['sender']
    )
    
    # Second split: separate validation from remaining data
    val_size_adjusted = validation_size / (1 - test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    # Create DataFrames
    train_df = pd.DataFrame({'content': X_train, 'sender': y_train})
    validation_df = pd.DataFrame({'content': X_validation, 'sender': y_validation})
    test_df = pd.DataFrame({'content': X_test, 'sender': y_test})
    
    # Save splits
    train_df.to_csv(TRAIN_DATA_FILE, index=False)
    validation_df.to_csv(VALIDATION_DATA_FILE, index=False)
    test_df.to_csv(TEST_DATA_FILE, index=False)
    
    print(f"✓ Splits created and saved:")
    print(f"  Training: {len(train_df):,} samples ({len(train_df)/len(emails_df)*100:.1f}%)")
    print(f"  Validation: {len(validation_df):,} samples ({len(validation_df)/len(emails_df)*100:.1f}%)")
    print(f"  Testing: {len(test_df):,} samples ({len(test_df)/len(emails_df)*100:.1f}%)")
    
    return train_df, validation_df, test_df


def train_baseline_naive_bayes(train_df, validation_df, test_df):
    """Train baseline Naive Bayes classifier."""
    print(f"\n{'='*60}")
    print("TRAINING BASELINE NAIVE BAYES CLASSIFIER")
    print(f"{'='*60}")
    
    print(f"Training samples: {len(train_df):,}")
    print(f"Validation samples: {len(validation_df):,}")
    print(f"Test samples: {len(test_df):,}")
    print(f"Number of classes: {len(train_df['sender'].unique())}")
    
    # Create pipeline with default parameters
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )),
        ('classifier', MultinomialNB())  # Default alpha=1.0
    ])
    
    # Train the model
    print("\nTraining model...")
    start_time = time.time()
    pipeline.fit(train_df['content'], train_df['sender'])
    training_time = time.time() - start_time
    print(f"✓ Training completed in {training_time:.2f} seconds")
    
    # Evaluate on validation set
    print("\n--- Validation Set Evaluation ---")
    val_pred = pipeline.predict(validation_df['content'])
    val_accuracy = accuracy_score(validation_df['sender'], val_pred)
    val_f1_weighted = f1_score(validation_df['sender'], val_pred, average='weighted')
    val_f1_macro = f1_score(validation_df['sender'], val_pred, average='macro')
    
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation F1 (weighted): {val_f1_weighted:.4f}")
    print(f"Validation F1 (macro): {val_f1_macro:.4f}")
    
    # Evaluate on test set
    print("\n--- Test Set Evaluation ---")
    test_pred = pipeline.predict(test_df['content'])
    test_accuracy = accuracy_score(test_df['sender'], test_pred)
    test_f1_weighted = f1_score(test_df['sender'], test_pred, average='weighted')
    test_f1_macro = f1_score(test_df['sender'], test_pred, average='macro')
    test_precision = precision_score(test_df['sender'], test_pred, average='weighted')
    test_recall = recall_score(test_df['sender'], test_pred, average='weighted')
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision (weighted): {test_precision:.4f}")
    print(f"Test Recall (weighted): {test_recall:.4f}")
    print(f"Test F1 (weighted): {test_f1_weighted:.4f}")
    print(f"Test F1 (macro): {test_f1_macro:.4f}")
    
    # Save model
    model_metadata = {
        'model_type': 'baseline_naive_bayes',
        'training_time': training_time,
        'validation_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'test_f1_weighted': test_f1_weighted,
        'test_f1_macro': test_f1_macro,
        'parameters': pipeline.get_params(),
        'timestamp': datetime.now().isoformat()
    }
    
    joblib.dump({'model': pipeline, 'metadata': model_metadata}, BASELINE_MODEL_FILE)
    print(f"✓ Model saved to {BASELINE_MODEL_FILE}")
    
    return pipeline, model_metadata


def optimize_naive_bayes(train_df, validation_df, test_df):
    """Train optimized Naive Bayes classifier with hyperparameter tuning."""
    print(f"\n{'='*60}")
    print("OPTIMIZING NAIVE BAYES HYPERPARAMETERS")
    print(f"{'='*60}")
    
    # Parameter grid for optimization
    param_grid = {
        'max_features': [3000, 5000, 7000],
        'min_df': [2, 3, 5],
        'max_df': [0.7, 0.8, 0.9],
        'alpha': [0.01, 0.1, 1.0, 10.0]
    }
    
    total_combinations = (len(param_grid['max_features']) * 
                         len(param_grid['min_df']) * 
                         len(param_grid['max_df']) * 
                         len(param_grid['alpha']))
    
    print(f"Testing {total_combinations} parameter combinations...")
    print("Parameters to optimize:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    # Manual grid search
    best_accuracy = 0
    best_params = {}
    best_model = None
    current_combination = 0
    
    print(f"\n{'Progress':<10} {'Parameters':<50} {'Val Acc':<10}")
    print("-" * 80)
    
    start_time = time.time()
    
    for max_features in param_grid['max_features']:
        for min_df in param_grid['min_df']:
            for max_df in param_grid['max_df']:
                for alpha in param_grid['alpha']:
                    current_combination += 1
                    
                    # Create and train pipeline
                    pipeline = Pipeline([
                        ('tfidf', TfidfVectorizer(
                            max_features=max_features,
                            min_df=min_df,
                            max_df=max_df,
                            stop_words='english'
                        )),
                        ('classifier', MultinomialNB(alpha=alpha))
                    ])
                    
                    # Train and validate
                    pipeline.fit(train_df['content'], train_df['sender'])
                    val_pred = pipeline.predict(validation_df['content'])
                    val_accuracy = accuracy_score(validation_df['sender'], val_pred)
                    
                    # Progress info
                    progress = f"{current_combination}/{total_combinations}"
                    params_str = f"feat={max_features}, min_df={min_df}, max_df={max_df}, α={alpha}"
                    print(f"{progress:<10} {params_str:<50} {val_accuracy:.4f}")
                    
                    # Track best model
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        best_params = {
                            'max_features': max_features,
                            'min_df': min_df,
                            'max_df': max_df,
                            'alpha': alpha
                        }
                        best_model = pipeline
    
    optimization_time = time.time() - start_time
    
    print(f"\n✓ Optimization completed in {optimization_time:.2f} seconds")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Final evaluation on test set
    print("\n--- Final Test Set Evaluation ---")
    test_pred = best_model.predict(test_df['content'])
    test_accuracy = accuracy_score(test_df['sender'], test_pred)
    test_f1_weighted = f1_score(test_df['sender'], test_pred, average='weighted')
    test_f1_macro = f1_score(test_df['sender'], test_pred, average='macro')
    test_precision = precision_score(test_df['sender'], test_pred, average='weighted')
    test_recall = recall_score(test_df['sender'], test_pred, average='weighted')
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision (weighted): {test_precision:.4f}")
    print(f"Test Recall (weighted): {test_recall:.4f}")
    print(f"Test F1 (weighted): {test_f1_weighted:.4f}")
    print(f"Test F1 (macro): {test_f1_macro:.4f}")
    
    # Save optimized model
    model_metadata = {
        'model_type': 'optimized_naive_bayes',
        'optimization_time': optimization_time,
        'best_validation_accuracy': best_accuracy,
        'test_accuracy': test_accuracy,
        'test_f1_weighted': test_f1_weighted,
        'test_f1_macro': test_f1_macro,
        'best_parameters': best_params,
        'total_combinations_tested': total_combinations,
        'timestamp': datetime.now().isoformat()
    }
    
    joblib.dump({'model': best_model, 'metadata': model_metadata}, OPTIMIZED_MODEL_FILE)
    print(f"✓ Optimized model saved to {OPTIMIZED_MODEL_FILE}")
    
    return best_model, model_metadata


def evaluate_models(baseline_metadata, optimized_metadata, test_df):
    """Compare baseline and optimized models."""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    # Load models for prediction examples
    baseline_data = joblib.load(BASELINE_MODEL_FILE)
    optimized_data = joblib.load(OPTIMIZED_MODEL_FILE)
    
    baseline_model = baseline_data['model']
    optimized_model = optimized_data['model']
    
    # Comparison table
    print(f"{'Metric':<25} {'Baseline':<12} {'Optimized':<12} {'Improvement':<12}")
    print("-" * 65)
    
    baseline_acc = baseline_metadata['test_accuracy']
    optimized_acc = optimized_metadata['test_accuracy']
    acc_improvement = optimized_acc - baseline_acc
    
    baseline_f1w = baseline_metadata['test_f1_weighted']
    optimized_f1w = optimized_metadata['test_f1_weighted']
    f1w_improvement = optimized_f1w - baseline_f1w
    
    baseline_f1m = baseline_metadata['test_f1_macro']
    optimized_f1m = optimized_metadata['test_f1_macro']
    f1m_improvement = optimized_f1m - baseline_f1m
    
    print(f"{'Accuracy':<25} {baseline_acc:<12.4f} {optimized_acc:<12.4f} {acc_improvement:>+11.4f}")
    print(f"{'F1-Score (weighted)':<25} {baseline_f1w:<12.4f} {optimized_f1w:<12.4f} {f1w_improvement:>+11.4f}")
    print(f"{'F1-Score (macro)':<25} {baseline_f1m:<12.4f} {optimized_f1m:<12.4f} {f1m_improvement:>+11.4f}")
    
    # Performance improvement percentage
    acc_improvement_pct = (acc_improvement / baseline_acc) * 100
    print(f"\n📈 Accuracy improvement: {acc_improvement_pct:+.1f}%")
    
    # Training time comparison
    baseline_time = baseline_metadata['training_time']
    optimization_time = optimized_metadata['optimization_time']
    print(f"⏱️  Baseline training time: {baseline_time:.2f} seconds")
    print(f"⏱️  Optimization time: {optimization_time:.2f} seconds")
    
    # Test a sample prediction
    sample_email = test_df.iloc[0]['content']
    actual_sender = test_df.iloc[0]['sender']
    
    baseline_pred = baseline_model.predict([sample_email])[0]
    baseline_proba = baseline_model.predict_proba([sample_email])[0].max()
    
    optimized_pred = optimized_model.predict([sample_email])[0]
    optimized_proba = optimized_model.predict_proba([sample_email])[0].max()
    
    print(f"\n📧 Sample Prediction:")
    print(f"Email snippet: {sample_email[:100]}...")
    print(f"Actual sender: {actual_sender}")
    print(f"Baseline prediction: {baseline_pred} (confidence: {baseline_proba:.4f})")
    print(f"Optimized prediction: {optimized_pred} (confidence: {optimized_proba:.4f})")
    
    # Save comparison results
    comparison_results = {
        'baseline': baseline_metadata,
        'optimized': optimized_metadata,
        'improvements': {
            'accuracy': acc_improvement,
            'f1_weighted': f1w_improvement,
            'f1_macro': f1m_improvement,
            'accuracy_percentage': acc_improvement_pct
        },
        'sample_prediction': {
            'email_snippet': sample_email[:200],
            'actual_sender': actual_sender,
            'baseline_prediction': baseline_pred,
            'baseline_confidence': float(baseline_proba),
            'optimized_prediction': optimized_pred,
            'optimized_confidence': float(optimized_proba)
        },
        'comparison_timestamp': datetime.now().isoformat()
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n✓ Detailed results saved to {RESULTS_FILE}")
    
    return comparison_results


def interactive_testing():
    """Allow interactive testing of the optimized model."""
    print(f"\n{'='*60}")
    print("INTERACTIVE TESTING")
    print(f"{'='*60}")
    
    # Load the optimized model
    try:
        optimized_data = joblib.load(OPTIMIZED_MODEL_FILE)
        model = optimized_data['model']
        print("✓ Optimized model loaded successfully")
    except FileNotFoundError:
        print("❌ Optimized model not found. Please train the model first.")
        return
    
    print("\nEnter email text to predict the sender (type 'quit' to exit)")
    print("-" * 60)
    
    while True:
        user_input = input("\n📧 Email text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            print("Please enter some email text.")
            continue
        
        try:
            # Make prediction
            prediction = model.predict([user_input])[0]
            confidence = model.predict_proba([user_input])[0].max()
            
            # Get top 3 predictions
            probabilities = model.predict_proba([user_input])[0]
            classes = model.classes_
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            
            print(f"\n🔮 Prediction: {prediction}")
            print(f"📊 Confidence: {confidence:.4f}")
            print(f"\nTop 3 candidates:")
            for i, idx in enumerate(top_3_indices, 1):
                sender = classes[idx]
                prob = probabilities[idx]
                print(f"  {i}. {sender}: {prob:.4f}")
                
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")


def show_existing_files():
    """Show information about existing files."""
    print("\n📁 Existing files:")
    
    # Check dataset files
    if os.path.exists(PROCESSED_DATA_FILE):
        df = pd.read_csv(PROCESSED_DATA_FILE)
        print(f"  ✓ Dataset: {PROCESSED_DATA_FILE} ({len(df):,} emails, {len(df['sender'].unique())} senders)")
    else:
        print(f"  ✗ Dataset: {PROCESSED_DATA_FILE} (not found)")
    
    # Check split files
    has_splits = (os.path.exists(TRAIN_DATA_FILE) and 
                  os.path.exists(VALIDATION_DATA_FILE) and 
                  os.path.exists(TEST_DATA_FILE))
    
    if has_splits:
        train_df = pd.read_csv(TRAIN_DATA_FILE)
        val_df = pd.read_csv(VALIDATION_DATA_FILE) 
        test_df = pd.read_csv(TEST_DATA_FILE)
        print(f"  ✓ Train/Val/Test splits: {len(train_df)}/{len(val_df)}/{len(test_df)} samples")
    else:
        print(f"  ✗ Train/Val/Test splits: not found")
    
    # Check model files
    if os.path.exists(BASELINE_MODEL_FILE):
        print(f"  ✓ Baseline model: {BASELINE_MODEL_FILE}")
    else:
        print(f"  ✗ Baseline model: {BASELINE_MODEL_FILE} (not found)")
    
    if os.path.exists(OPTIMIZED_MODEL_FILE):
        print(f"  ✓ Optimized model: {OPTIMIZED_MODEL_FILE}")
    else:
        print(f"  ✗ Optimized model: {OPTIMIZED_MODEL_FILE} (not found)")
    
    # Check results file
    if os.path.exists(RESULTS_FILE):
        print(f"  ✓ Results: {RESULTS_FILE}")
    else:
        print(f"  ✗ Results: {RESULTS_FILE} (not found)")


def remove_existing_files(file_patterns):
    """Remove existing files based on patterns."""
    removed_files = []
    
    for pattern in file_patterns:
        if pattern == "dataset" and os.path.exists(PROCESSED_DATA_FILE):
            os.remove(PROCESSED_DATA_FILE)
            removed_files.append(PROCESSED_DATA_FILE)
        elif pattern == "splits":
            for file_path in [TRAIN_DATA_FILE, VALIDATION_DATA_FILE, TEST_DATA_FILE]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    removed_files.append(file_path)
        elif pattern == "models":
            for file_path in [BASELINE_MODEL_FILE, OPTIMIZED_MODEL_FILE]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    removed_files.append(file_path)
        elif pattern == "results" and os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)
            removed_files.append(RESULTS_FILE)
    
    if removed_files:
        print(f"Removed files: {', '.join(removed_files)}")
    
    return removed_files


def main():
    """Main execution function."""
    print("="*60)
    print("SIMPLIFIED NAIVE BAYES EMAIL CLASSIFIER")
    print("="*60)
    
    # Check for maildir
    maildir_path = os.path.join(os.getcwd(), "maildir")
    if not os.path.exists(maildir_path):
        print(f"❌ Error: maildir folder not found at {maildir_path}")
        print("Please ensure the Enron email dataset 'maildir' folder exists in the current directory.")
        return
    
    # Show existing files
    show_existing_files()
    
    # Check what files exist
    has_dataset = os.path.exists(PROCESSED_DATA_FILE)
    has_splits = (os.path.exists(TRAIN_DATA_FILE) and 
                  os.path.exists(VALIDATION_DATA_FILE) and 
                  os.path.exists(TEST_DATA_FILE))
    has_models = (os.path.exists(BASELINE_MODEL_FILE) and 
                  os.path.exists(OPTIMIZED_MODEL_FILE))
    
    print(f"\n🎯 Available Options:")
    
    if has_models:
        print("1. Use existing models for interactive testing")
        print("2. Retrain models (with existing data)")
        print("3. Recreate splits and retrain models")
        print("4. Recreate dataset and retrain everything")
        print("5. Show model comparison results")
        choice = input("\nEnter your choice (1-5): ").strip()
    elif has_splits:
        print("1. Train models (with existing splits)")
        print("2. Recreate splits and train models")
        print("3. Recreate dataset and train everything")
        choice = input("\nEnter your choice (1-3): ").strip()
    elif has_dataset:
        print("1. Create splits and train models")
        print("2. Recreate dataset and train everything")
        choice = input("\nEnter your choice (1-2): ").strip()
    else:
        print("1. Create dataset and train everything")
        choice = input("\nEnter your choice (1): ").strip()
        if choice != "1":
            choice = "1"  # Force option 1 if no dataset exists
    
    try:
        force_recreate_dataset = False
        force_recreate_splits = False
        
        # Process user choice
        if has_models:
            if choice == "1":
                # Use existing models
                print("\n🎮 Loading existing models for interactive testing...")
                interactive_testing()
                return
            elif choice == "2":
                # Retrain models with existing data
                print("\n🔄 Retraining models with existing data...")
                pass  # Continue to training step
            elif choice == "3":
                # Recreate splits and retrain
                print("\n🔄 Recreating splits and retraining models...")
                remove_existing_files(["splits", "models", "results"])
                force_recreate_splits = True
            elif choice == "4":
                # Recreate everything
                print("\n🔄 Recreating dataset and retraining everything...")
                remove_existing_files(["dataset", "splits", "models", "results"])
                force_recreate_dataset = True
                force_recreate_splits = True
            elif choice == "5":
                # Show results
                if os.path.exists(RESULTS_FILE):
                    with open(RESULTS_FILE, 'r') as f:
                        results = json.load(f)
                    print("\n📊 Previous Model Comparison Results:")
                    print(f"Baseline Accuracy: {results['baseline']['test_accuracy']:.4f}")
                    print(f"Optimized Accuracy: {results['optimized']['test_accuracy']:.4f}")
                    print(f"Improvement: {results['improvements']['accuracy']:+.4f} ({results['improvements']['accuracy_percentage']:+.1f}%)")
                else:
                    print("❌ No results file found. Please train models first.")
                return
        elif has_splits:
            if choice == "2":
                # Recreate splits
                print("\n🔄 Recreating splits and training models...")
                remove_existing_files(["splits", "models", "results"])
                force_recreate_splits = True
            elif choice == "3":
                # Recreate everything
                print("\n🔄 Recreating dataset and training everything...")
                remove_existing_files(["dataset", "splits", "models", "results"])
                force_recreate_dataset = True
                force_recreate_splits = True
        elif has_dataset:
            if choice == "2":
                # Recreate dataset
                print("\n🔄 Recreating dataset and training everything...")
                remove_existing_files(["dataset", "splits", "models", "results"])
                force_recreate_dataset = True
                force_recreate_splits = True
        
        # Execute the pipeline
        print(f"\n{'='*60}")
        print("STARTING NAIVE BAYES PIPELINE")
        print(f"{'='*60}")
        
        # Step 1: Create dataset
        print("\n🔍 STEP 1: Creating dataset...")
        emails_df = create_dataset(maildir_path, force_recreate=force_recreate_dataset)
        
        # Step 2: Create train/test splits
        print("\n📊 STEP 2: Creating train/validation/test splits...")
        train_df, validation_df, test_df = create_train_test_split(emails_df, force_recreate=force_recreate_splits)
        
        # Step 3: Train baseline model
        print("\n🤖 STEP 3: Training baseline model...")
        baseline_model, baseline_metadata = train_baseline_naive_bayes(train_df, validation_df, test_df)
        
        # Step 4: Optimize hyperparameters
        print("\n⚡ STEP 4: Optimizing hyperparameters...")
        optimized_model, optimized_metadata = optimize_naive_bayes(train_df, validation_df, test_df)
        
        # Step 5: Compare models
        print("\n📋 STEP 5: Comparing models...")
        comparison_results = evaluate_models(baseline_metadata, optimized_metadata, test_df)
        
        # Step 6: Interactive testing (optional)
        print("\n🎮 STEP 6: Interactive testing...")
        response = input("Would you like to test the model interactively? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_testing()
        
        print(f"\n{'='*60}")
        print("✅ ALL STEPS COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"📁 Files created:")
        print(f"  - Dataset: {PROCESSED_DATA_FILE}")
        print(f"  - Train/val/test splits: {TRAIN_DATA_FILE}, {VALIDATION_DATA_FILE}, {TEST_DATA_FILE}")
        print(f"  - Models: {BASELINE_MODEL_FILE}, {OPTIMIZED_MODEL_FILE}")
        print(f"  - Results: {RESULTS_FILE}")
        print(f"\n🎯 Final optimized model accuracy: {optimized_metadata['test_accuracy']:.1%}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
