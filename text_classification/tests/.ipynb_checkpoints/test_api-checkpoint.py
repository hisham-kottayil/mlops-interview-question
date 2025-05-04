from fastapi.testclient import TestClient
import pytest
from src.text_classification import api

client = TestClient(api.app)

test_cases = [
    ("this is a positive example", "positive"),
    ("great content, really enjoyed it", "positive"),
    ("excellent work, very helpful", "positive"),
    ("this is a negative example", "negative"),
    ("terrible experience, waste of time", "negative"),
    ("poor quality, would not recommend", "negative"), 
    ("", "negative"),
    ("これはテストです", "negative")
]

test_sentence = "This is a positive sentence"

def test_null_input():
    input_data = {}
    response = client.post('/predict', json = input_data)
    assert response.status_code == 422, "Invalid input should return status code 422"

@pytest.mark.parametrize("sentence, expected_outcome", test_cases)
def test_valid_reponse(sentence, expected_outcome):
    response = client.post('/predict', json = {"sentence": sentence})
    assert response.status_code == 200, "Expected response status code 200"
    assert response.json()['prediction'] in ["positive", "negative"], "Expected response from ('positive', 'negative')" 


@pytest.mark.parametrize("sentence, expected_outcome", test_cases)
def test_response_correctness(sentence, expected_outcome):
    response = client.post('/predict', json = { "sentence": sentence})
    assert response.status_code == 200, "Expected response status code 200"
    assert response.json() == {"text": sentence, "prediction": expected_outcome}, "Wrong outcome"