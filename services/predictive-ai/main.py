from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def read_root():
    return {"message": "Welcome to the Predictive AI Service!"}

@app.post("/predict")
def predict():
    return {"failure_probability": 0.5, "risk_level": "Medium"}