import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os

# Define paths for model and vectorizer
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "text_classifier.joblib")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")

def train_and_save_model():
    """
    Trains a simple text classification model on dummy data
    and saves the model and vectorizer.
    """
    # Dummy data
    texts = [
        "this is a positive example",
        "great content, really enjoyed it",
        "excellent work, very helpful",
        "this is a negative example",
        "terrible experience, waste of time",
        "poor quality, would not recommend"
    ]
    labels = ["positive", "positive", "positive", "negative", "negative", "negative"]

    # Create a pipeline with TF-IDF Vectorizer and Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])

    
    # Train the model
    pipeline.fit(texts, labels)

    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save the fitted pipeline (includes vectorizer and classifier)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def load_model():
    """Loads the trained model pipeline from disk."""
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Training a new one.")
        train_and_save_model()

    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
def main():
    train_and_save_model()
    print("Model loaded")

if __name__ == "__main__":
    main()