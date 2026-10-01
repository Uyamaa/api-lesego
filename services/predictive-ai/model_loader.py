import joblib 

#Loaded both files at once, as the model alone is useless without the exact scaler it was trained with.
model = joblib.load("model/hdd_model.pkl")
scaler = joblib.load("model/hdd_scaler.pkl")    

def predict_failure(features):
    features_scaled = scaler.transform([features])  # Scale the features using the loaded scaler
    failure_probability = model.predict_proba(features_scaled)[0][1]  # Get the probability of failure (class 1)
    return failure_probability