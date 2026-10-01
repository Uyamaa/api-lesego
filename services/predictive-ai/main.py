from fastapi import FastAPI
from schemas import PredictRequest, PredictResponse
from model_loader import predict_failure, risk_level_from_probability



app = FastAPI()

#a simple automated status checker
@app.get("/health")
def read_root():
    return {"status": "ok"}

#this endpoint actively receives new data
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    
    return PredictResponse(
        drive_id=request.drive_id,
        failure_probability=0.5,
        risk_level="Medium"
    )