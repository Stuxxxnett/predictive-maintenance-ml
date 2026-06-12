from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Predictive Maintenance API")

model = joblib.load("models/rf_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

app.mount("/static", StaticFiles(directory="static"), name="static")

class SensorData(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    type: str

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/predict")
def predict(data: SensorData):
    input_df = pd.DataFrame([{
        "Air temperature [K]": data.air_temperature,
        "Process temperature [K]": data.process_temperature,
        "Rotational speed [rpm]": data.rotational_speed,
        "Torque [Nm]": data.torque,
        "Tool wear [min]": data.tool_wear,
        "Type": data.type
    }])

    input_df = pd.get_dummies(input_df, columns=["Type"], drop_first=True)

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "machine_failure_predicted": int(prediction),
        "failure_probability": round(float(probability), 4)
    }