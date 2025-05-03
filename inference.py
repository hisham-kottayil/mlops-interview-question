import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os
import typing

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


def predict_text(text: str, model) -> typing.Optional[str]:
    """
    Predicts the class of a given text using the loaded model.

    Args:
        text: The input text string.
        model: The loaded scikit-learn pipeline model.

    Returns:
        The predicted class label ("positive" or "negative") or None if prediction fails.
    """
    if model is None:
        print("Model is not loaded. Cannot predict.")
        return None
    try:
        # Predict uses the entire pipeline (vectorization + classification)
        prediction = model.predict([text])
        return prediction[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Example usage when the script is run directly
if __name__ == "__main__":
    # Ensure model exists or train it
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()

    # Load the model
    loaded_model = load_model()

    if loaded_model:
        # Example predictions
        test_text_positive = "this is fantastic content"
        test_text_negative = "this is very bad"

        prediction_pos = predict_text(test_text_positive, loaded_model)
        print(f"Prediction for '{test_text_positive}': {prediction_pos}")

        prediction_neg = predict_text(test_text_negative, loaded_model)
        print(f"Prediction for '{test_text_negative}': {prediction_neg}")

        # Example with new text
        new_text = "I am neutral about this."
        prediction_new = predict_text(new_text, loaded_model)
        print(f"Prediction for '{new_text}': {prediction_new}")


def main():
    # Load the model
    loaded_model = load_model()

    text = "I am Positive about this."
    prediction = predict_text(text, loaded_model)
    print(f"Prediction for '{text}': {prediction}")

if __name__ == "__main__":
    main()