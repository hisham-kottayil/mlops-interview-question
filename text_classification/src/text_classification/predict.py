import typing


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
