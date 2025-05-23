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
import os
import re

import joblib  # For saving/loading models
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Constants
DATA_DIR = "enron_data"
PROCESSED_DATA_FILE = "processed_emails.csv"
NUM_TOP_USERS = 150  # Number of unique users to include
MODEL_DIR = "models"  # Directory to store models
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

def build_naive_bayes_classifier(emails_df, save=True):
    """Build and evaluate a Naive Bayes classifier."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        emails_df['content'], 
        emails_df['sender'],
        test_size=0.2, 
        random_state=42,
        stratify=emails_df['sender']
    )
    
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
    pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model if requested
    if save:
        save_model(pipeline, BASELINE_MODEL_FILE, "baseline", accuracy, emails_df)
    
    return pipeline

def optimize_hyperparameters(emails_df, save=True):
    """Optimize hyperparameters using GridSearchCV."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        emails_df['content'], 
        emails_df['sender'],
        test_size=0.2, 
        random_state=42,
        stratify=emails_df['sender']
    )
    
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
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Print best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy with optimized model: {accuracy:.4f}")
    
    # Save the model if requested
    if save:
        save_model(best_model, OPTIMIZED_MODEL_FILE, "optimized", accuracy, emails_df)
    
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

def save_model(model, filepath, model_type, accuracy, emails_df):
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

def train_models(emails_df):
    """Train both baseline and optimized models."""
    # Train baseline model
    print("\n=== Training Baseline Model ===")
    baseline_model = build_naive_bayes_classifier(emails_df)
    
    # Optimize hyperparameters (optional, can be time-consuming)
    print("\n=== Optimizing Hyperparameters ===")
    optimized_model = optimize_hyperparameters(emails_df)
    
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
    
    if has_saved_models:
        # Display saved model information
        show_model_info()
        
        # Ask if user wants to use saved models
        print("Found previously trained models.")
        choice = input("Would you like to (1) use saved models or (2) retrain models? Enter 1 or 2: ")
        
        if choice == '1':
            # Load emails dataset for sample prediction
            emails_df = process_emails(maildir_path)
            
            # Load saved models
            print("\n=== Loading Saved Models ===")
            baseline_model = load_model(BASELINE_MODEL_FILE)
            optimized_model = load_model(OPTIMIZED_MODEL_FILE)
            
            if baseline_model is None or optimized_model is None:
                print("Error loading models. Will train new models.")
                baseline_model, optimized_model = train_models(emails_df)
        else:
            # Process emails and train models
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