from pydantic import BaseModel

# Defines what a client should send to the API for a prediction request
class PredictRequest(BaseModel):
    drive_id: int

# Defines what the API should return in response to a prediction request
class PredictResponse(BaseModel):
    drive_id: int
    failure_probability: float
    risk_level:str