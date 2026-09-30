from fastapi import FastAPI
from schemas import PredictRequest, PredictResponse

app = FastAPI()

@app.get("/health")
def read_root():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    # Placeholder logic - replace with actual prediction logic
    return PredictResponse(
        drive_id=request.drive_id,
        failure_probability=0.5,
        risk_level="Medium"
    )