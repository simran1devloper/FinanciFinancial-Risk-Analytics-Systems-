from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Initialize FastAPI app
app = FastAPI()

# Load the Keras model
model = load_model('model.h5')

# Load the preprocessing objects
with open('preprocess_final_file.pkl', 'rb') as file:
    preprocessing = pickle.load(file)

# Extract preprocessing objects
label_encoder_user_profile = preprocessing['label_encoder_user_profile']
minmax_scaler_blockchain = preprocessing['minmax_scaler_blockchain']
standard_scaler_behavioral = preprocessing['standard_scaler_behavioral']
robust_scaler_credit = preprocessing['robust_scaler_credit']

# Define the request body format
class PredictionRequest(BaseModel):
    KYCStatus: str
    AmountTransferred: float
    TransactionFrequency: float
    CreditScore: float

# Define the response format
class PredictionResponse(BaseModel):
    prediction: float

# Define the prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(data: PredictionRequest):
    try:
        # Preprocess the input data
        # Ensure KYCStatus is label encoded
        KYCStatus_encoded = label_encoder_user_profile.transform([data.KYCStatus])
        
        # Scale the numerical features
        AmountTransferred_scaled = minmax_scaler_blockchain.transform([[data.AmountTransferred]])
        TransactionFrequency_scaled = standard_scaler_behavioral.transform([[data.TransactionFrequency]])
        CreditScore_scaled = robust_scaler_credit.transform([[data.CreditScore]])

        # Combine preprocessed data
        X_preprocessed = np.hstack([
            KYCStatus_encoded.reshape(-1, 1),
            AmountTransferred_scaled,
            TransactionFrequency_scaled,
            CreditScore_scaled
        ])

        # Make prediction
        prediction = model.predict(X_preprocessed)

        # Return the prediction
        return PredictionResponse(prediction=float(prediction[0][0]))

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
