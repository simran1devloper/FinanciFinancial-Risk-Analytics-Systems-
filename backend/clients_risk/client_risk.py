# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import pickle
# import tensorflow as tf
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import numpy as np
#
# app = FastAPI()
#
# # Load the scaler and model
# with open('scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)
#
# model = tf.keras.models.load_model('model.h5')
#
# # Define the input data schema
# class RiskPredictionRequest(BaseModel):
#     loan_repayment: str
#     default_risk_pred: float
#     fraud_detected: int
#     trade_predictor: float
#
# # Define the prediction endpoint
# @app.post("/predict/")
# async def predict_risk(request: RiskPredictionRequest):
#     # Convert request data to DataFrame
#     input_data = pd.DataFrame([{
#         'loan_repayment': request.loan_repayment,
#         'default_risk_pred': request.default_risk_pred,
#         'fraud_detected': request.fraud_detected,
#         'trade_predictor': request.trade_predictor
#     }])
#
#     # Preprocess the input data
#     input_data['loan_repayment'] = input_data['loan_repayment'].map({'On-Time': 0, 'Late': 1, 'defaulted': 2})
#
#     # Ensure all necessary columns are present
#     expected_columns = ['loan_repayment', 'default_risk_pred', 'fraud_detected', 'trade_predictor']
#     missing_columns = [col for col in expected_columns if col not in input_data.columns]
#     if missing_columns:
#         raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing_columns)}")
#
#     # Scale the input data
#     X_scaled = scaler.transform(input_data)
#
#     # Make a prediction
#     prediction = model.predict(X_scaled)
#     client_risk = prediction[0][0].astype(float)
#
#     return {"client_risk": client_risk}
#
# # Run the FastAPI application
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8012)





from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify the origins here, e.g., ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load the scaler and model
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

model = tf.keras.models.load_model('model.h5')

# Define the input data schema
class RiskPredictionRequest(BaseModel):
    loan_repayment: str
    default_risk_pred: float
    fraud_detected: int
    trade_predictor: float

# Define the prediction endpoint
@app.post("/predict/")
async def predict_risk(request: RiskPredictionRequest):
    # Convert request data to DataFrame
    input_data = pd.DataFrame([{
        'loan_repayment': request.loan_repayment,
        'default_risk_pred': request.default_risk_pred,
        'fraud_detected': request.fraud_detected,
        'trade_predictor': request.trade_predictor
    }])

    # Preprocess the input data
    input_data['loan_repayment'] = input_data['loan_repayment'].map({'On-Time': 0, 'Late': 1, 'defaulted': 2})

    # Ensure all necessary columns are present
    expected_columns = ['loan_repayment', 'default_risk_pred', 'fraud_detected', 'trade_predictor']
    missing_columns = [col for col in expected_columns if col not in input_data.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing_columns)}")

    # Scale the input data
    X_scaled = scaler.transform(input_data)

    # Make a prediction
    prediction = model.predict(X_scaled)
    client_risk = prediction[0][0].astype(float)

    return {"client_risk": client_risk}

# Run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8012)
