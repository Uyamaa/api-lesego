from fastapi import FastAPI
from schemas import PredictRequest, PredictResponse
from model_loader import predict_failure, risk_level_from_probability
from databaseCN import get_latest_reading

app = FastAPI()

#a simple automated status checker
@app.get("/health")
def db_ok = check_connection()
    model_ok = model is not None and scaler is not None
    healthy = db_ok and model_ok

    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "ok" if healthy else "degraded",
            "database": "up" if db_ok else "down",
            "model": "loaded" if model_ok else "missing",
        },
    )

#this endpoint actively receives new data
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    row = get_latest_reading(request.drive_id)

    if row is None:
        raise HTTPException(status_code=404, detail="Drive ID not found in the database.")

     # Assuming the first column is drive_id and the rest are features
    features = list(row[1:]) 
    
    failure_probability = predict_failure(features)
    risk_level = risk_level_from_probability(failure_probability)
    
    
    return PredictResponse(
        drive_id=request.drive_id,
        failure_probability=failure_probability,
        risk_level=risk_level
    )
