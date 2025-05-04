import os
import shutil
import joblib
from src.text_classification import train

def cleanup():
    model_path = train.MODEL_DIR
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

def test_model_creation_train_and_save_model():
    cleanup()
    train.train_and_save_model()
    model_path = train.MODEL_DIR

    assert os.path.exists(model_path), "Model is not created"

def train_model_creation_load_model():
    cleanup()
    train.load_model()
    model_path = train.MODEL_DIR

    assert os.path.exists(model_path), "Model is not created"

def train_model_type():
    from sklearn.pipeline import Pipeline
    train.train_and_save_model()
    model = train.load_model()
    assert  isinstance(model, Pipeline), "Model is not an sklearn pipeline"








    