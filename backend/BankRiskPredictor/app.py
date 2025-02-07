# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import numpy as np
#
# # Initialize FastAPI app
# app = FastAPI()
#
# # Load the model
# model = joblib.load('bank_risk_model.pkl')
#
# # Define the input format using Pydantic BaseModel
# class RiskInput(BaseModel):
#     Market: bool
#     OPR: bool
#     Trade: bool
#     Fraud: float
#     Var1: float
#     Var2: bool
#     EAD1: bool
#     EAD2: float
#     Loan_Amount: float
#
# # Define POST route for prediction
# @app.post("/predict/")
# def predict_risk(data: RiskInput):
#     try:
#         # Prepare the data for prediction
#         input_data = np.array([[
#             data.Market,
#             data.OPR,
#             data.Trade,
#             data.Fraud,
#             data.Var1,
#             data.Var2,
#             data.EAD1,
#             data.EAD2,
#             data.Loan_Amount
#         ]])
#
#         # Make a prediction
#         prediction = model.predict(input_data)
#
#         # Apply the scaling formula as before
#         prediction_scaled = (prediction / (1 + np.abs(prediction))) * 100
#
#         # Ensure the output is positive and convert to a Python float
#         prediction_scaled = float(np.where(prediction_scaled < 0, 0, prediction_scaled)[0])
#
#         # Apply round to the prediction
#         return {"risk_percentage": round(prediction_scaled, 4)}
#
#     except Exception as e:
#         return {"error": str(e)}
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, or specify specific origins here
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load the model
model = joblib.load('bank_risk_model.pkl')

# Define the input format using Pydantic BaseModel
class RiskInput(BaseModel):
    Market: bool
    OPR: bool
    Trade: bool
    Fraud: float
    Var1: float
    Var2: bool
    EAD1: bool
    EAD2: float
    Loan_Amount: float

# Define POST route for prediction
@app.post("/predict/")
def predict_risk(data: RiskInput):
    try:
        # Prepare the data for prediction
        input_data = np.array([[
            data.Market,
            data.OPR,
            data.Trade,
            data.Fraud,
            data.Var1,
            data.Var2,
            data.EAD1,
            data.EAD2,
            data.Loan_Amount
        ]])

        # Make a prediction
        prediction = model.predict(input_data)

        # Apply the scaling formula as before
        prediction_scaled = (prediction / (1 + np.abs(prediction))) * 100

        # Ensure the output is positive and convert to a Python float
        prediction_scaled = float(np.where(prediction_scaled < 0, 0, prediction_scaled)[0])

        # Apply round to the prediction
        return {"risk_percentage": round(prediction_scaled, 4)}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009)
