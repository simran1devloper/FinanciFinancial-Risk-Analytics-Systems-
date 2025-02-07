# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
#
# app = FastAPI()
#
# # Load the operational risk model
# model = joblib.load('opr_risk_model.pkl')
#
# # Define the request body schema
# class PredictionRequest(BaseModel):
#     features: list
#
# # Define the prediction endpoint for operational risk
# @app.post('/predict_operational_risk')
# def predict_operational_risk(request: PredictionRequest):
#     features = request.features
#     prediction = model.predict([features])
#     return {'prediction': prediction.tolist()}

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

# Load the operational risk model
model = joblib.load('opr_risk_model.pkl')

# Define the request body schema
class PredictionRequest(BaseModel):
    features: list

# Add CORS middleware to allow requests from your React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React development server
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define the prediction endpoint for operational risk
@app.post('/predict_operational_risk')
def predict_operational_risk(request: PredictionRequest):
    features = request.features
    prediction = model.predict([features])
    return {'prediction': prediction.tolist()}

# Run the FastAPI application
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8001)
