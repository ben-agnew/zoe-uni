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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# Try to import TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM classifier will be skipped.")
    print("To enable LSTM, install TensorFlow: pip install tensorflow")

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")  # Set default style
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: Matplotlib and/or Seaborn not available. Visualizations will be skipped.")
    print("To enable visualizations, install the required packages:")
    print("pip install matplotlib seaborn")

# Set matplotlib parameters for better plots if available
if VISUALIZATION_AVAILABLE:
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9

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
LSTM_MODEL_FILE = os.path.join(MODEL_DIR, "lstm_model.h5")
RANDOM_FOREST_MODEL_FILE = os.path.join(MODEL_DIR, "random_forest_model.joblib")
LOGISTIC_REGRESSION_MODEL_FILE = os.path.join(MODEL_DIR, "logistic_regression_model.joblib")
NEURAL_NETWORK_MODEL_FILE = os.path.join(MODEL_DIR, "neural_network_model.joblib")
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
    # Display dataset information
    print("=== Naive Bayes Training Setup ===")
    print(f"Training samples: {len(train_df):,}")
    print(f"Algorithm: Multinomial Naive Bayes (fast training)")
    
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
    print("\n=== Training Naive Bayes ===")
    print("Computing class probabilities...")
    start_time = time.time()
    pipeline.fit(train_df['content'], train_df['sender'])
    training_time = time.time() - start_time
    print(f"✓ Naive Bayes training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    print("\n=== Naive Bayes Model Evaluation ===")
    y_pred = pipeline.predict(test_df['content'])
    accuracy = accuracy_score(test_df['sender'], y_pred)
    print(f"✅ Accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(test_df['sender'], y_pred))
    
    # Save the model if requested
    if save:
        # Add training time to metadata
        metadata_extras = {'training_time': training_time}
        save_model(pipeline, BASELINE_MODEL_FILE, "baseline", accuracy, train_df, metadata_extras)
    
    return pipeline

def build_lstm_classifier(train_df, test_df, save=True):
    """Build and evaluate an LSTM classifier."""
    if not TENSORFLOW_AVAILABLE:
        print("Error: TensorFlow not available. Please install TensorFlow to use LSTM classifier.")
        print("Install with: pip install tensorflow")
        return None
    
    # Display dataset information
    print("=== LSTM Classifier Training Setup ===")
    print(f"Training samples: {len(train_df):,}")
    print(f"Testing samples: {len(test_df):,}")
    print(f"Number of unique senders: {len(train_df['sender'].unique())}")
    print(f"Average email length: {train_df['content'].str.len().mean():.0f} characters")
    
    # Prepare data
    print("\n=== Preparing Data for LSTM ===")
    X_train = train_df['content'].values
    X_test = test_df['content'].values
    y_train = train_df['sender'].values
    y_test = test_df['sender'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_train_categorical = to_categorical(y_train_encoded)
    y_test_categorical = to_categorical(y_test_encoded)
    
    # Tokenize text
    max_features = 10000  # Maximum number of words to keep
    max_length = 100      # Maximum sequence length
    
    tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')
    
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Sequence length: {max_length}")
    
    # Build LSTM model
    print("\n=== Building LSTM Model ===")
    model = Sequential([
        Embedding(input_dim=max_features, output_dim=128, input_length=max_length),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model architecture:")
    model.summary()
    
    # Train the model
    print("\n=== Starting LSTM Training ===")
    print("🚀 Training LSTM classifier...")
    print("-" * 50)
    
    start_time = time.time()
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train with validation split
    history = model.fit(
        X_train_padded, y_train_categorical,
        batch_size=32,
        epochs=10,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"✓ LSTM Training completed successfully!")
    print(f"⏱️  Total training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
    
    # Show memory usage if psutil is available
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"💾 Memory usage: {memory_mb:.1f} MB")
    except ImportError:
        pass
    print(f"{'='*50}")
    
    # Evaluate the model
    print("\n=== LSTM Model Evaluation ===")
    y_pred_proba = model.predict(X_test_padded)
    y_pred_encoded = np.argmax(y_pred_proba, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create a wrapper class to mimic sklearn interface
    class LSTMWrapper:
        def __init__(self, model, tokenizer, label_encoder, max_length):
            self.model = model
            self.tokenizer = tokenizer
            self.label_encoder = label_encoder
            self.max_length = max_length
        
        def predict(self, texts):
            sequences = self.tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
            predictions = self.model.predict(padded)
            predicted_classes = np.argmax(predictions, axis=1)
            return self.label_encoder.inverse_transform(predicted_classes)
        
        def predict_proba(self, texts):
            sequences = self.tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
            return self.model.predict(padded)
    
    wrapper = LSTMWrapper(model, tokenizer, label_encoder, max_length)
    
    # Save the model if requested
    if save:
        save_lstm_model(wrapper, LSTM_MODEL_FILE, accuracy, train_df, training_time)
    
    return wrapper

def build_fast_lstm_classifier(train_df, test_df, save=True):
    """Build a fast LSTM classifier with reduced complexity."""
    if not TENSORFLOW_AVAILABLE:
        print("Error: TensorFlow not available. Please install TensorFlow to use LSTM classifier.")
        return None
    
    print("=== Fast LSTM Training Setup ===")
    
    # Prepare data with reduced parameters for faster training
    X_train = train_df['content'].values
    X_test = test_df['content'].values
    y_train = train_df['sender'].values
    y_test = test_df['sender'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    y_train_categorical = to_categorical(y_train_encoded)
    y_test_categorical = to_categorical(y_test_encoded)
    
    # Reduced tokenization parameters
    max_features = 5000  # Reduced vocabulary
    max_length = 50      # Shorter sequences
    
    tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')
    
    # Build simpler LSTM model
    model = Sequential([
        Embedding(input_dim=max_features, output_dim=64, input_length=max_length),
        LSTM(32, dropout=0.2),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training Fast LSTM...")
    start_time = time.time()
    
    # Train with fewer epochs
    model.fit(
        X_train_padded, y_train_categorical,
        batch_size=64,  # Larger batch size
        epochs=5,       # Fewer epochs
        validation_split=0.1,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"✓ Fast LSTM completed in {training_time:.2f} seconds")
    
    # Evaluate
    y_pred_proba = model.predict(X_test_padded)
    y_pred_encoded = np.argmax(y_pred_proba, axis=1)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {accuracy:.4f}")
    
    # Create wrapper
    class LSTMWrapper:
        def __init__(self, model, tokenizer, label_encoder, max_length):
            self.model = model
            self.tokenizer = tokenizer
            self.label_encoder = label_encoder
            self.max_length = max_length
        
        def predict(self, texts):
            sequences = self.tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
            predictions = self.model.predict(padded)
            predicted_classes = np.argmax(predictions, axis=1)
            return self.label_encoder.inverse_transform(predicted_classes)
        
        def predict_proba(self, texts):
            sequences = self.tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
            return self.model.predict(padded)
    
    wrapper = LSTMWrapper(model, tokenizer, label_encoder, max_length)
    
    if save:
        save_lstm_model(wrapper, LSTM_MODEL_FILE, accuracy, train_df, training_time)
    
    return wrapper

def build_random_forest_classifier(train_df, test_df, save=True):
    """Build and evaluate a Random Forest classifier."""
    # Display dataset information
    print("=== Random Forest Training Setup ===")
    print(f"Training samples: {len(train_df):,}")
    print(f"Forest configuration: 100 trees with parallel processing")
    
    # Create a pipeline with TF-IDF and Random Forest
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )),
        ('classifier', RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1,
            verbose=1  # Enable verbose output
        ))
    ])
    
    # Train the model
    print("\n=== Training Random Forest ===")
    print("Building 100 decision trees in parallel...")
    start_time = time.time()
    pipeline.fit(train_df['content'], train_df['sender'])
    training_time = time.time() - start_time
    print(f"✓ Random Forest training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    print("\n=== Random Forest Model Evaluation ===")
    y_pred = pipeline.predict(test_df['content'])
    accuracy = accuracy_score(test_df['sender'], y_pred)
    print(f"✅ Accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(test_df['sender'], y_pred))
    
    # Save the model if requested
    if save:
        metadata_extras = {'training_time': training_time}
        save_model(pipeline, RANDOM_FOREST_MODEL_FILE, "random_forest", accuracy, train_df, metadata_extras)
    
    return pipeline

def build_logistic_regression_classifier(train_df, test_df, save=True):
    """Build and evaluate a Logistic Regression classifier."""
    # Display dataset information
    print("=== Logistic Regression Training Setup ===")
    print(f"Training samples: {len(train_df):,}")
    print(f"Max iterations: 1000 (with parallel processing)")
    
    # Create a pipeline with TF-IDF and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )),
        ('classifier', LogisticRegression(
            max_iter=1000, 
            random_state=42, 
            n_jobs=-1,
            verbose=1  # Enable verbose output
        ))
    ])
    
    # Train the model
    print("\n=== Training Logistic Regression ===")
    print("Optimizing coefficients using L-BFGS solver...")
    start_time = time.time()
    pipeline.fit(train_df['content'], train_df['sender'])
    training_time = time.time() - start_time
    print(f"✓ Logistic Regression training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    print("\n=== Logistic Regression Model Evaluation ===")
    y_pred = pipeline.predict(test_df['content'])
    accuracy = accuracy_score(test_df['sender'], y_pred)
    print(f"✅ Accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(test_df['sender'], y_pred))
    
    # Save the model if requested
    if save:
        metadata_extras = {'training_time': training_time}
        save_model(pipeline, LOGISTIC_REGRESSION_MODEL_FILE, "logistic_regression", accuracy, train_df, metadata_extras)
    
    return pipeline

def build_neural_network_classifier(train_df, test_df, save=True):
    """Build and evaluate a Neural Network (MLP) classifier."""
    # Display dataset information
    print("=== Neural Network Training Setup ===")
    print(f"Training samples: {len(train_df):,}")
    print(f"Architecture: Input -> TF-IDF(5000) -> Hidden(100,50) -> Output({len(train_df['sender'].unique())})")
    
    # Create a pipeline with TF-IDF and Multi-layer Perceptron
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=True  # Enable verbose output for training progress
        ))
    ])
    
    # Train the model
    print("\n=== Starting Neural Network Training ===")
    print("Training with early stopping enabled...")
    print("Convergence progress will be shown below:")
    print("-" * 50)
    
    start_time = time.time()
    pipeline.fit(train_df['content'], train_df['sender'])
    training_time = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"✓ Neural Network training completed!")
    print(f"⏱️  Training time: {training_time:.2f} seconds")
    
    # Get training info from the classifier
    mlp_classifier = pipeline.named_steps['classifier']
    print(f"🔄 Training iterations: {mlp_classifier.n_iter_}")
    print(f"📉 Final loss: {mlp_classifier.loss_:.6f}")
    print(f"{'='*50}")
    
    # Evaluate the model
    print("\n=== Neural Network Model Evaluation ===")
    y_pred = pipeline.predict(test_df['content'])
    accuracy = accuracy_score(test_df['sender'], y_pred)
    print(f"✅ Accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(test_df['sender'], y_pred))
    
    # Save the model if requested
    if save:
        metadata_extras = {
            'training_time': training_time,
            'n_iterations': int(mlp_classifier.n_iter_),
            'final_loss': float(mlp_classifier.loss_),
            'hidden_layers': mlp_classifier.hidden_layer_sizes
        }
        save_model(pipeline, NEURAL_NETWORK_MODEL_FILE, "neural_network", accuracy, train_df, metadata_extras)
    
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
    print("\n=== Hyperparameter Optimization ===")
    print("Testing combinations of parameters:")
    print("  • max_features: [3000, 5000]")
    print("  • min_df: [2, 3]") 
    print("  • max_df: [0.7, 0.8]")
    print("  • alpha: [0.01, 0.1, 1.0]")
    print(f"Total combinations: {len(param_grid['vectorizer__max_features']) * len(param_grid['vectorizer__min_df']) * len(param_grid['vectorizer__max_df']) * len(param_grid['classifier__alpha'])}")
    print("Progress will be shown below...")
    print("-" * 50)
    
    start_time = time.time()
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)  # Increased verbose level
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

def save_lstm_model(lstm_wrapper, filepath, accuracy, train_df, training_time):
    """Save an LSTM model and its components."""
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available, cannot save LSTM model")
        return
    
    import pickle

    # Save the Keras model
    lstm_wrapper.model.save(filepath)
    
    # Save tokenizer and label encoder
    tokenizer_path = filepath.replace('.h5', '_tokenizer.pkl')
    encoder_path = filepath.replace('.h5', '_encoder.pkl')
    metadata_path = filepath.replace('.h5', '_metadata.pkl')
    
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(lstm_wrapper.tokenizer, f)
    
    with open(encoder_path, 'wb') as f:
        pickle.dump(lstm_wrapper.label_encoder, f)
    
    # Save metadata including max_length and training info
    metadata = {
        'max_length': lstm_wrapper.max_length,
        'accuracy': accuracy,
        'training_time': training_time,
        'num_samples': len(train_df)
    }
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"LSTM model saved to {filepath}")
    print(f"Tokenizer saved to {tokenizer_path}")
    print(f"Label encoder saved to {encoder_path}")
    print(f"Metadata saved to {metadata_path}")

def load_lstm_model(filepath):
    """Load an LSTM model and its components."""
    import pickle
    
    if not os.path.exists(filepath):
        return None
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available, cannot load LSTM model")
        return None
    
    print(f"Loading LSTM model from {filepath}")
    
    # Load the Keras model
    model = tf.keras.models.load_model(filepath)
    
    # Load tokenizer and label encoder
    tokenizer_path = filepath.replace('.h5', '_tokenizer.pkl')
    encoder_path = filepath.replace('.h5', '_encoder.pkl')
    
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Get max_length from metadata or use default
    metadata_path = filepath.replace('.h5', '_metadata.pkl')
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        max_length = metadata.get('max_length', 100)
    except:
        max_length = 100
    
    # Create wrapper
    class LSTMWrapper:
        def __init__(self, model, tokenizer, label_encoder, max_length):
            self.model = model
            self.tokenizer = tokenizer
            self.label_encoder = label_encoder
            self.max_length = max_length
        
        def predict(self, texts):
            sequences = self.tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
            predictions = self.model.predict(padded)
            predicted_classes = np.argmax(predictions, axis=1)
            return self.label_encoder.inverse_transform(predicted_classes)
        
        def predict_proba(self, texts):
            sequences = self.tokenizer.texts_to_sequences(texts)
            padded = pad_sequences(sequences, maxlen=self.max_length, padding='post', truncating='post')
            return self.model.predict(padded)
    
    return LSTMWrapper(model, tokenizer, label_encoder, max_length)

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

def train_all_models(emails_df):
    """Train all classifier models."""
    # Get or create train/test split
    train_df, test_df = get_or_create_train_test_split(emails_df)
    
    print("\n" + "="*80)
    print("🚀 TRAINING ALL CLASSIFIER MODELS")
    print("="*80)
    print(f"📊 Dataset Summary:")
    print(f"   • Training samples: {len(train_df):,}")
    print(f"   • Testing samples: {len(test_df):,}")
    print(f"   • Unique senders: {len(train_df['sender'].unique())}")
    print(f"   • Total training time estimate: 10-30 minutes (depending on hardware)")
    print("="*80)
    
    models = {}
    total_models = 6  # Total number of models to train
    model_times = {}
    overall_start_time = time.time()
    
    # Train baseline Naive Bayes model
    print(f"\n[1/{total_models}] 🧠 Training Baseline Naive Bayes Model")
    start_time = time.time()
    models['naive_bayes'] = build_naive_bayes_classifier(train_df, test_df)
    model_times['naive_bayes'] = time.time() - start_time
    
    # Train LSTM model (replaces SVM)
    print(f"\n[2/{total_models}] 🧠 Training LSTM Model (This may take a while)")
    start_time = time.time()
    models['lstm'] = build_fast_lstm_classifier(train_df, test_df)
    model_times['lstm'] = time.time() - start_time
    
    # Train Random Forest model
    print(f"\n[3/{total_models}] 🌲 Training Random Forest Model")
    start_time = time.time()
    models['random_forest'] = build_random_forest_classifier(train_df, test_df)
    model_times['random_forest'] = time.time() - start_time
    
    # Train Logistic Regression model
    print(f"\n[4/{total_models}] 📈 Training Logistic Regression Model")
    start_time = time.time()
    models['logistic_regression'] = build_logistic_regression_classifier(train_df, test_df)
    model_times['logistic_regression'] = time.time() - start_time
    
    # Train Neural Network model
    print(f"\n[5/{total_models}] 🧪 Training Neural Network Model")
    start_time = time.time()
    models['neural_network'] = build_neural_network_classifier(train_df, test_df)
    model_times['neural_network'] = time.time() - start_time
    
    # Optimize hyperparameters (optional, can be time-consuming)
    print(f"\n[6/{total_models}] 🔧 Optimizing Hyperparameters for Naive Bayes")
    start_time = time.time()
    models['optimized_naive_bayes'] = optimize_hyperparameters(train_df, test_df)
    model_times['optimized_naive_bayes'] = time.time() - start_time
    
    # Training summary
    total_time = time.time() - overall_start_time
    print("\n" + "="*80)
    print("🎉 ALL MODELS TRAINING COMPLETED!")
    print("="*80)
    print(f"⏱️  Total training time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print("\n📊 Individual model training times:")
    for model_name, training_time in model_times.items():
        percentage = (training_time / total_time) * 100
        print(f"   • {model_name.replace('_', ' ').title()}: {training_time:.2f}s ({percentage:.1f}%)")
    
    print(f"\n✅ Successfully trained {len(models)} models!")
    print("💡 Use option 6 from the main menu to compare all models.")
    print("="*80)
    
    return models

def train_models(emails_df):
    """Train both baseline and optimized models (legacy function for backward compatibility)."""
    # Get or create train/test split
    train_df, test_df = get_or_create_train_test_split(emails_df)
    
    # Train baseline model
    print("\n=== Training Baseline Model ===")
    baseline_model = build_naive_bayes_classifier(train_df, test_df)
    
    # Optimize hyperparameters (optional, can be time-consuming)
    print("\n=== Optimizing Hyperparameters ===")
    optimized_model = optimize_hyperparameters(train_df, test_df)
    
    return baseline_model, optimized_model

def load_all_models():
    """Load all trained models."""
    models = {}
    model_files = {
        'naive_bayes': BASELINE_MODEL_FILE,
        'optimized_naive_bayes': OPTIMIZED_MODEL_FILE,
        'lstm': LSTM_MODEL_FILE,
        'random_forest': RANDOM_FOREST_MODEL_FILE,
        'logistic_regression': LOGISTIC_REGRESSION_MODEL_FILE,
        'neural_network': NEURAL_NETWORK_MODEL_FILE
    }
    
    for name, filepath in model_files.items():
        if name == 'lstm':
            model = load_lstm_model(filepath)
        else:
            model = load_model(filepath)
        if model is not None:
            models[name] = model
            print(f"Loaded {name.replace('_', ' ').title()} model")
    
    return models

def compare_all_models():
    """Compare performance of all trained models."""
    print("\n" + "="*60)
    print("COMPARING ALL CLASSIFIER MODELS")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)
    
    # Load all available models
    models = load_all_models()
    
    if not models:
        print("No trained models found.")
        return None
    
    # Evaluate each model
    all_results = {}
    for name, model in models.items():
        try:
            results, y_true, y_pred, y_proba = evaluate_model_comprehensive(
                model, test_df, name.replace('_', ' ').title())
            all_results[name] = results
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
    
    if all_results:
        # Create comparison summary
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        # Create comparison DataFrame
        comparison_data = []
        for name, results in all_results.items():
            comparison_data.append({
                'Model': name.replace('_', ' ').title(),
                'Accuracy': f"{results['accuracy']:.4f}",
                'F1-Score (Weighted)': f"{results['f1_weighted']:.4f}",
                'F1-Score (Macro)': f"{results['f1_macro']:.4f}",
                'Precision (Weighted)': f"{results['precision_weighted']:.4f}",
                'Recall (Weighted)': f"{results['recall_weighted']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best performing models
        best_accuracy = max(all_results.values(), key=lambda x: x['accuracy'])
        best_f1_weighted = max(all_results.values(), key=lambda x: x['f1_weighted'])
        best_f1_macro = max(all_results.values(), key=lambda x: x['f1_macro'])
        
        print(f"\nBest Accuracy: {best_accuracy['model_name']} ({best_accuracy['accuracy']:.4f})")
        print(f"Best F1-Score (Weighted): {best_f1_weighted['model_name']} ({best_f1_weighted['f1_weighted']:.4f})")
        print(f"Best F1-Score (Macro): {best_f1_macro['model_name']} ({best_f1_macro['f1_macro']:.4f})")
        
        # Create comprehensive visualization dashboard
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Get model metadata for training time comparison
        model_metadata = get_model_metadata()
        
        if VISUALIZATION_AVAILABLE:
            print("\n🎨 Creating comprehensive visualization dashboard...")
            generated_plots = create_model_performance_dashboard(
                all_results, models, test_df, model_metadata, RESULTS_DIR
            )
            print(f"✅ Generated {len(generated_plots)} visualization plots!")
        
        # Export results
        csv_path = os.path.join(RESULTS_DIR, f"all_models_comparison_{timestamp}.csv")
        comparison_df.to_csv(csv_path, index=False)
        print(f"\nComparison results saved to: {csv_path}")
        
        # Export detailed results
        detailed_results = {
            'evaluation_timestamp': timestamp,
            'dataset_info': {
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'num_classes': len(train_df['sender'].unique()),
                'top_senders': train_df['sender'].value_counts().head(10).to_dict()
            },
            'model_results': all_results,
            'best_models': {
                'best_accuracy': best_accuracy['model_name'],
                'best_f1_weighted': best_f1_weighted['model_name'],
                'best_f1_macro': best_f1_macro['model_name']
            }
        }
        
        json_path = os.path.join(RESULTS_DIR, f"all_models_detailed_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        print(f"Detailed results saved to: {json_path}")
        
        # Print summary of generated visualizations
        if VISUALIZATION_AVAILABLE and 'generated_plots' in locals():
            print(f"\n📊 VISUALIZATION SUMMARY:")
            for i, plot_path in enumerate(generated_plots, 1):
                plot_name = os.path.basename(plot_path).replace(f"_{timestamp}.png", "").replace("0" + str(i) + "_", "").replace("_", " ").title()
                print(f"  {i}. {plot_name}: {plot_path}")
        else:
            print(f"\n📊 No visualizations generated (matplotlib/seaborn not available)")
    
    return all_results

def create_multi_model_comparison_plot(all_results, save_path=None):
    """Create a comprehensive comparison plot for multiple models."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Skipping multi-model comparison plot.")
        return None
    
    if not all_results:
        print("No results to plot.")
        return None
        
    # Prepare data for plotting
    models = list(all_results.keys())
    model_names = [name.replace('_', ' ').title() for name in models]
    
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'f1_macro']
    metric_names = ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)', 'F1-Score (Macro)']
    
    # Create subplot for each metric
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes_flat[i]
        
        values = [all_results[model][metric] for model in models]
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        bars = ax.bar(range(len(models)), values, color=colors, alpha=0.8)
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title(metric_name)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Create overall ranking plot in the last subplot
    ax = axes_flat[5]
    
    # Calculate average performance across all metrics
    avg_scores = []
    for model in models:
        avg_score = np.mean([all_results[model][metric] for metric in metrics])
        avg_scores.append(avg_score)
    
    # Sort models by average score
    sorted_data = sorted(zip(model_names, avg_scores), key=lambda x: x[1], reverse=True)
    sorted_names, sorted_scores = zip(*sorted_data)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_names)))
    bars = ax.bar(range(len(sorted_names)), sorted_scores, color=colors, alpha=0.8)
    ax.set_xlabel('Models')
    ax.set_ylabel('Average Score')
    ax.set_title('Overall Model Ranking (Average Score)')
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=45, ha='right')
    ax.set_ylim(0, 1)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{score:.3f}\n(#{i+1})', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-model comparison plot saved to {save_path}")
    
    return plt

def create_confusion_matrix_comparison(all_results, test_df, models, save_path=None):
    """Create confusion matrix comparison for all models."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Skipping confusion matrix comparison.")
        return None
    
    n_models = len(models)
    if n_models == 0:
        return None
    
    # Calculate grid size
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    fig.suptitle('Confusion Matrix Comparison Across Models', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if n_models > 1 else [axes]
    else:
        axes = axes.flatten()
    
    unique_senders = sorted(test_df['sender'].unique())
    
    for i, (model_name, model) in enumerate(models.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Get predictions
        y_true = test_df['sender']
        y_pred = model.predict(test_df['content'])
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=unique_senders)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'{model_name.replace("_", " ").title()}')
        
        # Only show labels for first few classes to avoid clutter
        if len(unique_senders) <= 20:
            tick_marks = np.arange(len(unique_senders))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(unique_senders, rotation=45, ha='right')
            ax.set_yticklabels(unique_senders)
        else:
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix comparison saved to {save_path}")
    
    return plt

def create_training_time_comparison(model_metadata, save_path=None):
    """Create training time comparison visualization."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Skipping training time comparison.")
        return None
    
    if not model_metadata:
        print("No model metadata available for training time comparison.")
        return None
    
    # Extract training times
    models_with_time = {}
    for model_name, metadata in model_metadata.items():
        if 'training_time' in metadata:
            models_with_time[model_name] = metadata['training_time']
    
    if not models_with_time:
        print("No training time data available.")
        return None
    
    # Sort by training time
    sorted_models = sorted(models_with_time.items(), key=lambda x: x[1])
    model_names, training_times = zip(*sorted_models)
    
    # Convert to readable format
    model_display_names = [name.replace('_', ' ').title() for name in model_names]
    time_minutes = [time/60 for time in training_times]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color gradient based on training time
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(model_names)))
    bars = ax.barh(range(len(model_names)), time_minutes, color=colors, alpha=0.8)
    
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_display_names)
    ax.set_xlabel('Training Time (minutes)')
    ax.set_title('Model Training Time Comparison', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, time_min, time_sec) in enumerate(zip(bars, time_minutes, training_times)):
        width = bar.get_width()
        if time_sec < 60:
            label = f'{time_sec:.1f}s'
        else:
            label = f'{time_min:.1f}m'
        ax.text(width + max(time_minutes) * 0.01, bar.get_y() + bar.get_height()/2,
                label, ha='left', va='center', fontweight='bold')
    
    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training time comparison saved to {save_path}")
    
    return plt

def create_radar_chart_comparison(all_results, save_path=None):
    """Create radar chart comparison of model performance."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Skipping radar chart comparison.")
        return None
    
    if not all_results:
        return None
    
    # Metrics for radar chart
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'f1_macro']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Weighted', 'F1-Macro']
    
    # Number of metrics
    N = len(metrics)
    
    # Compute angle for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    fig.suptitle('Model Performance Radar Chart Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    # Colors for different models
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_results)))
    
    # Plot each model
    for i, (model_name, results) in enumerate(all_results.items()):
        values = [results[metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        model_display_name = model_name.replace('_', ' ').title()
        ax.plot(angles, values, 'o-', linewidth=2, label=model_display_name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Add metric labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # Add grid lines
    ax.grid(True)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_rlabel_position(0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Radar chart comparison saved to {save_path}")
    
    return plt

def create_performance_distribution_plot(all_results, save_path=None):
    """Create distribution plots showing performance spread across metrics."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Skipping performance distribution plot.")
        return None
    
    if not all_results:
        return None
    
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'f1_macro']
    metric_labels = ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)', 'F1-Score (Macro)']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Performance Distribution Analysis Across Models', fontsize=16, fontweight='bold')
    axes_flat = axes.flatten()
    
    # Box plot for each metric
    for i, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        ax = axes_flat[i]
        
        # Collect values for this metric across all models
        model_names = []
        values = []
        
        for model_name, results in all_results.items():
            model_names.append(model_name.replace('_', ' ').title())
            values.append(results[metric])
        
        # Create box plot (even though we have single values, this shows distribution)
        bp = ax.boxplot([values], patch_artist=True, labels=['All Models'])
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_alpha(0.7)
        
        # Overlay individual points
        y_pos = np.ones(len(values))
        colors = plt.cm.Set3(np.linspace(0, 1, len(values)))
        scatter = ax.scatter(y_pos, values, c=colors, s=100, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Add model labels
        for j, (name, value) in enumerate(zip(model_names, values)):
            ax.annotate(name, (1, value), xytext=(1.1, value), 
                       fontsize=9, ha='left', va='center')
        
        ax.set_title(metric_label)
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    # Performance summary in the last subplot
    ax = axes_flat[5]
    
    # Calculate overall rankings
    model_rankings = {}
    for model_name in all_results.keys():
        total_score = sum(all_results[model_name][metric] for metric in metrics)
        model_rankings[model_name] = total_score / len(metrics)
    
    # Sort by ranking
    sorted_rankings = sorted(model_rankings.items(), key=lambda x: x[1], reverse=True)
    names, scores = zip(*sorted_rankings)
    
    display_names = [name.replace('_', ' ').title() for name in names]
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    
    bars = ax.bar(range(len(names)), scores, color=colors, alpha=0.8)
    ax.set_xlabel('Models')
    ax.set_ylabel('Average Score')
    ax.set_title('Overall Model Ranking')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    
    # Add value labels and rankings
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{score:.3f}\n#{i+1}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance distribution plot saved to {save_path}")
    
    return plt

def create_metric_correlation_heatmap(all_results, save_path=None):
    """Create correlation heatmap between different performance metrics."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Skipping metric correlation heatmap.")
        return None
    
    if not all_results:
        return None
    
    # Extract metrics data
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'f1_macro']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Weighted', 'F1-Macro']
    
    # Create DataFrame with all metrics
    data = []
    model_names = []
    for model_name, results in all_results.items():
        row = [results[metric] for metric in metrics]
        data.append(row)
        model_names.append(model_name.replace('_', ' ').title())
    
    df = pd.DataFrame(data, columns=metric_labels, index=model_names)
    
    # Check if we have enough data points for meaningful correlation
    if len(data) < 3:
        print(f"Warning: Only {len(data)} models available. Correlation analysis needs at least 3 data points.")
        print("Creating a simplified metric comparison instead...")
        
        # Create a simple comparison plot instead
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Metric values by model
        df_transposed = df.T
        im1 = ax1.imshow(df_transposed.values, cmap='viridis', aspect='auto')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_yticks(range(len(metric_labels)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.set_yticklabels(metric_labels)
        ax1.set_title('Model Performance Across Metrics', fontweight='bold')
        
        # Add values to cells
        for i in range(len(metric_labels)):
            for j in range(len(model_names)):
                text = ax1.text(j, i, f'{df_transposed.iloc[i, j]:.3f}',
                               ha="center", va="center", color="white", fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, label='Performance Score')
        
        # Right plot: Metric comparison bar chart
        metric_means = df.mean()
        metric_stds = df.std()
        
        bars = ax2.bar(range(len(metric_labels)), metric_means, 
                      yerr=metric_stds, capsize=5, alpha=0.7, 
                      color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax2.set_xticks(range(len(metric_labels)))
        ax2.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax2.set_ylabel('Average Performance Score')
        ax2.set_title('Average Performance by Metric (with std dev)', fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean_val, std_val in zip(bars, metric_means, metric_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                    f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
    else:
        # Calculate correlation matrix
        correlation_matrix = df.corr()
        
        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Left plot: Correlation heatmap
        # Use a better colormap with proper scaling
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)  # Mask upper triangle
        
        if 'sns' in globals():
            sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                       square=True, ax=ax1, cbar_kws={"shrink": .8})
        else:
            # Manual heatmap without seaborn
            masked_corr = correlation_matrix.copy()
            masked_corr[mask] = np.nan  # Set upper triangle to NaN
            
            im1 = ax1.imshow(masked_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            ax1.set_xticks(range(len(metric_labels)))
            ax1.set_yticks(range(len(metric_labels)))
            ax1.set_xticklabels(metric_labels, rotation=45, ha='right')
            ax1.set_yticklabels(metric_labels)
            
            # Add correlation values to visible cells
            for i in range(len(metric_labels)):
                for j in range(len(metric_labels)):
                    if not mask[i, j]:  # Only show lower triangle
                        value = correlation_matrix.iloc[i, j]
                        color = 'white' if abs(value) > 0.5 else 'black'
                        ax1.text(j, i, f'{value:.2f}', ha="center", va="center", 
                               color=color, fontweight='bold')
            
            plt.colorbar(im1, ax=ax1, label='Correlation Coefficient')
        
        ax1.set_title('Metric Correlation Matrix\n(Lower Triangle Only)', fontweight='bold')
        
        # Right plot: Scatter plot matrix for most correlated metrics
        # Find the pair with highest absolute correlation (excluding diagonal)
        corr_copy = correlation_matrix.copy()
        np.fill_diagonal(corr_copy.values, 0)  # Remove diagonal
        max_corr_idx = np.unravel_index(np.argmax(np.abs(corr_copy.values)), corr_copy.shape)
        metric1, metric2 = metric_labels[max_corr_idx[0]], metric_labels[max_corr_idx[1]]
        
        x_vals = df[metric1]
        y_vals = df[metric2]
        
        # Create scatter plot
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_names)))
        for i, (x, y, name) in enumerate(zip(x_vals, y_vals, model_names)):
            ax2.scatter(x, y, c=[colors[i]], s=100, alpha=0.7, edgecolors='black')
            ax2.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, alpha=0.8)
        
        # Add trend line
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(x_vals), max(x_vals), 100)
        ax2.plot(x_trend, p(x_trend), 'r--', alpha=0.8, linewidth=2)
        
        ax2.set_xlabel(metric1)
        ax2.set_ylabel(metric2)
        ax2.set_title(f'Highest Correlation: {metric1} vs {metric2}\n(r = {corr_copy.iloc[max_corr_idx]:.3f})', 
                     fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.suptitle('Performance Metrics Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metric analysis dashboard saved to {save_path}")
    
    return plt

def create_model_performance_dashboard(all_results, models, test_df, model_metadata=None, save_dir=None):
    """Create a comprehensive dashboard with multiple visualizations."""
    if not VISUALIZATION_AVAILABLE:
        print("Visualization libraries not available. Skipping performance dashboard.")
        return []
    
    if not all_results:
        print("No results available for dashboard creation.")
        return []
    
    if save_dir is None:
        save_dir = RESULTS_DIR
    
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    generated_plots = []
    
    print("\n🎨 Creating comprehensive visualization dashboard...")
    
    # 1. Multi-model comparison (existing)
    print("📊 Generating multi-model comparison chart...")
    plot_path = os.path.join(save_dir, f"01_multi_model_comparison_{timestamp}.png")
    plt_obj = create_multi_model_comparison_plot(all_results, plot_path)
    if plt_obj:
        plt_obj.close()
        generated_plots.append(plot_path)
    
    # 2. Confusion matrix comparison
    print("🔍 Generating confusion matrix comparison...")
    plot_path = os.path.join(save_dir, f"02_confusion_matrix_comparison_{timestamp}.png")
    plt_obj = create_confusion_matrix_comparison(all_results, test_df, models, plot_path)
    if plt_obj:
        plt_obj.close()
        generated_plots.append(plot_path)
    
    # 3. Training time comparison
    if model_metadata:
        print("⏱️ Generating training time comparison...")
        plot_path = os.path.join(save_dir, f"03_training_time_comparison_{timestamp}.png")
        plt_obj = create_training_time_comparison(model_metadata, plot_path)
        if plt_obj:
            plt_obj.close()
            generated_plots.append(plot_path)
    
    # 4. Radar chart comparison
    print("🕸️ Generating radar chart comparison...")
    plot_path = os.path.join(save_dir, f"04_radar_chart_comparison_{timestamp}.png")
    plt_obj = create_radar_chart_comparison(all_results, plot_path)
    if plt_obj:
        plt_obj.close()
        generated_plots.append(plot_path)
    
    # 5. Performance distribution plot
    print("📈 Generating performance distribution analysis...")
    plot_path = os.path.join(save_dir, f"05_performance_distribution_{timestamp}.png")
    plt_obj = create_performance_distribution_plot(all_results, plot_path)
    if plt_obj:
        plt_obj.close()
        generated_plots.append(plot_path)
    
    # 6. Metric correlation heatmap
    print("📊 Generating metrics analysis dashboard...")
    plot_path = os.path.join(save_dir, f"06_metrics_analysis_dashboard_{timestamp}.png")
    plt_obj = create_metric_correlation_heatmap(all_results, plot_path)
    if plt_obj:
        plt_obj.close()
        generated_plots.append(plot_path)
    
    print(f"\n✅ Dashboard generation completed! Generated {len(generated_plots)} visualizations.")
    print(f"📁 All plots saved in: {save_dir}")
    
    return generated_plots

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
            choice = input("Would you like to:\n"
                          "(1) use saved models\n"
                          "(2) retrain models with existing split\n"
                          "(3) recreate split and retrain\n"
                          "(4) generate evaluation report\n"
                          "(5) train all classifier types\n"
                          "(6) compare all trained models\n"
                          "Enter 1, 2, 3, 4, 5, or 6: ")
        else:
            choice = input("Would you like to:\n"
                          "(1) use saved models\n"
                          "(2) retrain models\n"
                          "(3) generate evaluation report\n"
                          "(4) train all classifier types\n"
                          "(5) compare all trained models\n"
                          "Enter 1, 2, 3, 4, or 5: ")
        
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
        elif choice == '5' or (choice == '4' and not has_saved_splits):
            # Train all classifier types
            print(f"Using existing maildir data at: {maildir_path}")
            emails_df = process_emails(maildir_path)
            all_models = train_all_models(emails_df)
            
            # Use the best performing model for interactive testing
            print("\nTraining completed! You can now use option 6 to compare all models.")
            response = input("Would you like to compare all models now? (y/n): ")
            if response.lower() == 'y':
                compare_all_models()
            
            # Use optimized naive bayes for interactive testing
            optimized_model = all_models.get('optimized_naive_bayes', all_models.get('naive_bayes'))
            emails_df = process_emails(maildir_path)
            
        elif choice == '6' or (choice == '5' and not has_saved_splits):
            # Compare all trained models
            results = compare_all_models()
            if results:
                print("\nModel comparison completed!")
                response = input("Would you like to continue with interactive testing? (y/n): ")
                if response.lower() != 'y':
                    return
                
                # Load data and use best model for testing
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