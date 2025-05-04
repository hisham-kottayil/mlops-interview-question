from src.text_classification import predict, train
import pytest
import os
import shutil 

test_sentence = "This is a good sentence"

test_cases = [
    ("this is a positive example", "positive"),
    ("great content, really enjoyed it", "positive"),
    ("excellent work, very helpful", "positive"),
    ("this is a negative example", "negative"),
    ("terrible experience, waste of time", "negative"),
    ("poor quality, would not recommend", "negative")  
]

def setup_module():
    # Clean up any previous model dir before tests
    if os.path.exists(train.MODEL_DIR):
        shutil.rmtree(train.MODEL_DIR)
        
def test_none_case():
    model = None
    assert predict.predict_text(test_sentence, model) == None, "Should return None when model is None"

def test_model_type():
    train.train_and_save_model()
    model = train.load_model()
    from sklearn.pipeline import Pipeline

    assert isinstance(model, Pipeline), "Loaded model is not an sklearn pipeline"

def test_return_value_type():
    prediction_possibilities = ["positive", "negative"]
    train.train_and_save_model()
    model = train.load_model()

    prediction = str(predict.predict_text(test_sentence, model))
    assert type(prediction)==str, "Prediction should be a string" 

def test_integer_input():
    train.train_and_save_model()
    model = train.load_model()
    test_sentence = 123
    prediction = predict.predict_text(test_sentence, model)
    assert prediction is None, "Should return None for non-string input"

@pytest.mark.parametrize("sentence, expected_prediction", test_cases)
def test_return_values(sentence, expected_prediction):
    train.train_and_save_model()
    model = train.load_model()
    prediction = str(predict.predict_text(sentence, model))
    
    assert prediction == expected_prediction, f"Expected {expected_prediction}, but got {prediction}. Sentence: {sentence}"

    