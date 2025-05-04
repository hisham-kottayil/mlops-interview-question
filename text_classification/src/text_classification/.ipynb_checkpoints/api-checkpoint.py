from fastapi import FastAPI, HTTPException
from .train import load_model
from .predict import predict_text
from pydantic import BaseModel

class inputData(BaseModel):
    sentence: str

model = load_model()
app = FastAPI()

@app.post('/predict')
def predict(body: inputData):

    prediction = predict_text(body.sentence, model)
    # prediction = None
    if prediction is None:
        raise HTTPException(status_code = 500, detail = "Prediction failed")
    return {
        "text": body.sentence,
        "prediction": prediction
    }
