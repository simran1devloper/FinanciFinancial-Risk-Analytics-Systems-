# from fastapi import FastAPI
# from pydantic import BaseModel
# import joblib
# import numpy as np
# import pandas as pd
#
# # Initialize the FastAPI app
# app = FastAPI()
#
# # Load the trained Random Forest model and the scaler
# model = joblib.load('new_forest.pkl')
# scaler = joblib.load('new_scaler.pkl')
#
# # Define the columns that should be scaled (from the original code)
# columns_to_scale = ['age', 'income', 'credit_score', 'credit_utilization',
#                     'loan_amount', 'loan_term', 'interest_rate', 'dti_ratio',
#                     'savings', 'investments', 'monthly_expenses', 'collateral_value']
#
# # Define input structure for prediction using Pydantic
# class LoanData(BaseModel):
#     age: int
#     dependents: int
#     work_experience: int
#     income: float
#     credit_score: int
#     credit_utilization: float
#     open_credit_accounts: int
#     loan_amount: float
#     loan_term: int
#     interest_rate: float
#     dti_ratio: float
#     savings: float
#     investments: float
#     monthly_expenses: float
#     collateral_value: float
#
# # Health check endpoint
# @app.get("/")
# def read_root():
#     return {"message": "Loan Repayment Prediction API is running."}
#
# # Prediction endpoint
# @app.post("/predict")
# def predict(data: LoanData):
#     # Convert input data to a pandas DataFrame
#     input_data = pd.DataFrame([data.dict()])
#
#     # Scale the input data (using the columns to scale)
#     input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])
#
#     # Make prediction using the trained model
#     prediction = model.predict(input_data)
#
#     # Convert the prediction result to human-readable format
#     repayment_status = {0: "Defaulted", 1: "Late", 2: "On-Time"}
#     result = repayment_status[prediction[0]]
#
#     return {"repayment_status": result}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Initialize the FastAPI app
app = FastAPI()

# Allow CORS for your React app running on localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load the trained Random Forest model and the scaler
model = joblib.load('new_forest.pkl')
scaler = joblib.load('new_scaler.pkl')

# Define the columns that should be scaled (from the original code)
columns_to_scale = ['age', 'income', 'credit_score', 'credit_utilization',
                    'loan_amount', 'loan_term', 'interest_rate', 'dti_ratio',
                    'savings', 'investments', 'monthly_expenses', 'collateral_value']

# Define input structure for prediction using Pydantic
class LoanData(BaseModel):
    age: int
    dependents: int
    work_experience: int
    income: float
    credit_score: int
    credit_utilization: float
    open_credit_accounts: int
    loan_amount: float
    loan_term: int
    interest_rate: float
    dti_ratio: float
    savings: float
    investments: float
    monthly_expenses: float
    collateral_value: float

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Loan Repayment Prediction API is running."}

# Prediction endpoint
@app.post("/predict")
def predict(data: LoanData):
    # Convert input data to a pandas DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Scale the input data (using the columns to scale)
    input_data[columns_to_scale] = scaler.transform(input_data[columns_to_scale])

    # Make prediction using the trained model
    prediction = model.predict(input_data)

    # Convert the prediction result to human-readable format
    repayment_status = {0: "Defaulted", 1: "Late", 2: "On-Time"}
    result = repayment_status[prediction[0]]

    return {"repayment_status": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
