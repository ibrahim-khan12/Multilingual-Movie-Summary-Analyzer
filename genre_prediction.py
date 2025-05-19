import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class MovieGenrePredictor:
    def __init__(self):
        """Initialize the genre predictor"""
        self.vectorizer = None
        self.model = None
        self.mlb = None
        self.genres = None
    
    def load_data(self, train_file, test_file=None):
        """
        Load training and test data
        
        Args:
            train_file (str): Path to the training data file
            test_file (str): Path to the test data file (optional)
            
        Returns:
            tuple: (train_df, test_df)
        """
        train_df = pd.read_csv(train_file)
        
        # Convert string representation of genres to list
        train_df['genres'] = train_df['genres'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
        
        if test_file and os.path.exists(test_file):
            test_df = pd.read_csv(test_file)
            test_df['genres'] = test_df['genres'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
        else:
            test_df = None
        
        return train_df, test_df
    
    def preprocess_data(self, train_df, test_df=None):
        """
        Preprocess the data for training/testing
        
        Args:
            train_df (DataFrame): Training data
            test_df (DataFrame): Test data (optional)
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        # Clean genre names by removing curly braces
        train_df['genres'] = train_df['genres'].apply(
            lambda genres: [g.strip().replace('}', '') for g in genres]
        )
        if test_df is not None:
            test_df['genres'] = test_df['genres'].apply(
                lambda genres: [g.strip().replace('}', '') for g in genres]
            )
        
        # Extract features using TF-IDF
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train = self.vectorizer.fit_transform(train_df['summary'])
        
        # Convert genres to multi-hot encoding
        self.mlb = MultiLabelBinarizer()
        y_train = self.mlb.fit_transform(train_df['genres'])
        
        # Store genre names
        self.genres = self.mlb.classes_
        
        if test_df is not None:
            X_test = self.vectorizer.transform(test_df['summary'])
            y_test = self.mlb.transform(test_df['genres'])
        else:
            X_test, y_test = None, None
        
        return X_train, y_train, X_test, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train the genre prediction model
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            model: Trained model
        """
        base_model = LogisticRegression(max_iter=1000, C=1.0)
        self.model = MultiOutputClassifier(base_model)
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Performance metrics
        """
        if X_test is None or y_test is None:
            return None
        
        # Predict on test data
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print("Model Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, X_test, y_test):
        """
        Plot confusion matrix for each genre
        
        Args:
            X_test: Test features
            y_test: Test labels
        """
        if X_test is None or y_test is None:
            return
        
        y_pred = self.model.predict(X_test)
        
        # For each genre, plot a confusion matrix
        plt.figure(figsize=(15, 15))
        
        # Limit to top 9 genres for visualization
        top_genres = min(9, len(self.genres))
        
        for i in range(top_genres):
            plt.subplot(3, 3, i + 1)
            cm = confusion_matrix(y_test[:, i], y_pred[:, i])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix: {self.genres[i]}')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png')
        plt.close()
    
    def save_model(self, model_dir='models'):
        """
        Save the trained model and associated components
        
        Args:
            model_dir (str): Directory to save model files
        """
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.vectorizer, f'{model_dir}/vectorizer.pkl')
        joblib.dump(self.model, f'{model_dir}/model.pkl')
        joblib.dump(self.mlb, f'{model_dir}/multilabel_binarizer.pkl')
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir='models'):
        """
        Load a saved model
        
        Args:
            model_dir (str): Directory containing model files
        """
        self.vectorizer = joblib.load(f'{model_dir}/vectorizer.pkl')
        self.model = joblib.load(f'{model_dir}/model.pkl')
        self.mlb = joblib.load(f'{model_dir}/multilabel_binarizer.pkl')
        self.genres = self.mlb.classes_
        
        print(f"Model loaded from {model_dir}")
    
    def predict_genres(self, summary_text):
        """
        Predict genres for a given movie summary
        
        Args:
            summary_text (str): Movie summary text
            
        Returns:
            list: Predicted genres
        """
        if self.vectorizer is None or self.model is None or self.mlb is None:
            raise ValueError("Model not trained or loaded")
        
        # Process the input text
        X = self.vectorizer.transform([summary_text])
        
        # Make prediction with probabilities without displaying them
        y_pred_proba = self.model.predict_proba(X)
        
        # Use a threshold for prediction
        threshold = 0.1  # Use a low threshold to increase recall
        
        # Get prediction using threshold
        y_pred = np.zeros((1, len(self.genres)))
        for i, estimator in enumerate(self.model.estimators_):
            prob = estimator.predict_proba(X)[0][1]  # Probability of positive class
            y_pred[0, i] = 1 if prob > threshold else 0
        
        # Convert binary predictions back to genre labels
        predicted_genres = self.mlb.inverse_transform(y_pred)[0]
        
        # If no genres were predicted, return the top 2 highest probability genres
        if len(predicted_genres) == 0:
            top_indices = np.argsort([est.predict_proba(X)[0][1] for est in self.model.estimators_])[-2:]
            predicted_genres = [self.genres[i] for i in top_indices]
            print("\nUsing top 2 highest probability genres")
        
        return list(predicted_genres)

if __name__ == "__main__":
    predictor = MovieGenrePredictor()
    
    # Load data
    train_df, test_df = predictor.load_data('train_data.csv', 'test_data.csv')
    
    # Preprocess data
    X_train, y_train, X_test, y_test = predictor.preprocess_data(train_df, test_df)
    
    # Train model
    predictor.train_model(X_train, y_train)
    
    # Evaluate model
    metrics = predictor.evaluate_model(X_test, y_test)
    
    # Plot confusion matrix
    predictor.plot_confusion_matrix(X_test, y_test)
    
    # Save model
    predictor.save_model()
    
    # Test prediction
    sample_summary = "A team of scientists explore an uncharted island in the Pacific, venturing into the domain of the mighty Kong, and must fight to escape a primal Eden."
    predicted_genres = predictor.predict_genres(sample_summary)
    print(f"Predicted genres for sample summary: {predicted_genres}")
