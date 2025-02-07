# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
#
# app = FastAPI()
#
# # Load the market risk model
# model = joblib.load('model.pkl')
#
# # Define the request body schema
# class PredictionRequest(BaseModel):
#     features: list
#
# # Define the prediction endpoint for market risk
# @app.post('/predict_market_risk')
# def predict_market_risk(request: PredictionRequest):
#     features = request.features
#     prediction = model.predict([features])
#     return {'prediction': prediction.tolist()}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# Allow CORS from localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your React frontend's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load the market risk model
model = joblib.load('model.pkl')

# Define the request body schema
class PredictionRequest(BaseModel):
    features: list

# Define the prediction endpoint for market risk
@app.post('/predict_market_risk')
def predict_market_risk(request: PredictionRequest):
    features = request.features
    prediction = model.predict([features])
    return {'prediction': prediction.tolist()}

