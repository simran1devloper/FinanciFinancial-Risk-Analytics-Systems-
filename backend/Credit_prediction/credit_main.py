# #credit_main.py
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import joblib
# import numpy as np
#
# # Initialize FastAPI app
# app = FastAPI()
#
# # Load the pre-trained model and scaler
# model = joblib.load("credit_model.pkl")
# scaler = joblib.load("scaler.pkl")
#
# # Define the input data structure using Pydantic
# class CreditScoringRequest(BaseModel):
#     DerogCnt: float
#     CollectCnt: float
#     BanruptcyInd: float
#     InqCnt06: float
#     InqTimeLast: float
#     InqFinanceCnt24: float
#     TLTimeFirst: float
#     TLTimeLast: float
#     TLCnt03: float
#     TLCnt12: float
#     TLCnt24: float
#     TLCnt: float
#     TLSum: float
#     TLMaxSum: float
#     TLSatCnt: float
#     TLDel60Cnt: float
#     TLBadCnt24: float
#     TL75UtilCnt: float
#     TL50UtilCnt: float
#     TLBalHCPct: float
#     TLSatPct: float
#     TLDel3060Cnt24: float
#     TLDel90Cnt24: float
#     TLDel60CntAll: float
#     TLOpenPct: float
#     TLBadDerogCnt: float
#     TLDel60Cnt24: float
#     TLOpen24Pct: float
#
# # Define the response data structure
# class CreditScoringResponse(BaseModel):
#     predicted_outcome: int
#     probability_0: float
#     probability_1: float
#
# # Test route
# @app.get("/")
# def read_root():
#     return {"message": "Welcome to the Credit Scoring API!"}
#
# # Prediction route
# @app.post("/predict", response_model=CreditScoringResponse)
# def predict_credit_scoring(request: CreditScoringRequest):
#     try:
#         # Extract the features from the request
#         features = [
#             request.DerogCnt, request.CollectCnt, request.BanruptcyInd, request.InqCnt06,
#             request.InqTimeLast, request.InqFinanceCnt24, request.TLTimeFirst, request.TLTimeLast,
#             request.TLCnt03, request.TLCnt12, request.TLCnt24, request.TLCnt, request.TLSum,
#             request.TLMaxSum, request.TLSatCnt, request.TLDel60Cnt, request.TLBadCnt24, request.TL75UtilCnt,
#             request.TL50UtilCnt, request.TLBalHCPct, request.TLSatPct, request.TLDel3060Cnt24,
#             request.TLDel90Cnt24, request.TLDel60CntAll, request.TLOpenPct, request.TLBadDerogCnt,
#             request.TLDel60Cnt24, request.TLOpen24Pct
#         ]
#
#         # Convert the features into a NumPy array and reshape for scaling
#         features = np.array(features).reshape(1, -1)
#
#         # Scale the features using the loaded scaler
#         features_scaled = scaler.transform(features)
#
#         # Make predictions using the model
#         prediction = model.predict(features_scaled)
#         prediction_proba = model.predict_proba(features_scaled)
#
#         # Return the prediction and probabilities
#         return CreditScoringResponse(
#             predicted_outcome=int(prediction[0]),
#             probability_0=prediction_proba[0][0],
#             probability_1=prediction_proba[0][1]
#         )
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this to match the origin of your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pre-trained model and scaler
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")


# Define the input data structure using Pydantic
class CreditScoringRequest(BaseModel):
    DerogCnt: float
    CollectCnt: float
    BanruptcyInd: float
    InqCnt06: float
    InqTimeLast: float
    InqFinanceCnt24: float
    TLTimeFirst: float
    TLTimeLast: float
    TLCnt03: float
    TLCnt12: float
    TLCnt24: float
    TLCnt: float
    TLSum: float
    TLMaxSum: float
    TLSatCnt: float
    TLDel60Cnt: float
    TLBadCnt24: float
    TL75UtilCnt: float
    TL50UtilCnt: float
    TLBalHCPct: float
    TLSatPct: float
    TLDel3060Cnt24: float
    TLDel90Cnt24: float
    TLDel60CntAll: float
    TLOpenPct: float
    TLBadDerogCnt: float
    TLDel60Cnt24: float
    TLOpen24Pct: float


# Define the response data structure
class CreditScoringResponse(BaseModel):
    predicted_outcome: int
    probability_0: float
    probability_1: float


# Test route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Credit Scoring API!"}


# Prediction route
@app.post("/predict", response_model=CreditScoringResponse)
def predict_credit_scoring(request: CreditScoringRequest):
    try:
        # Extract the features from the request
        features = [
            request.DerogCnt, request.CollectCnt, request.BanruptcyInd, request.InqCnt06,
            request.InqTimeLast, request.InqFinanceCnt24, request.TLTimeFirst, request.TLTimeLast,
            request.TLCnt03, request.TLCnt12, request.TLCnt24, request.TLCnt, request.TLSum,
            request.TLMaxSum, request.TLSatCnt, request.TLDel60Cnt, request.TLBadCnt24, request.TL75UtilCnt,
            request.TL50UtilCnt, request.TLBalHCPct, request.TLSatPct, request.TLDel3060Cnt24,
            request.TLDel90Cnt24, request.TLDel60CntAll, request.TLOpenPct, request.TLBadDerogCnt,
            request.TLDel60Cnt24, request.TLOpen24Pct
        ]

        # Convert the features into a NumPy array and reshape for scaling
        features = np.array(features).reshape(1, -1)

        # Scale the features using the loaded scaler
        features_scaled = scaler.transform(features)

        # Make predictions using the model
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)

        # Return the prediction and probabilities
        return CreditScoringResponse(
            predicted_outcome=int(prediction[0]),
            probability_0=prediction_proba[0][0],
            probability_1=prediction_proba[0][1]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
