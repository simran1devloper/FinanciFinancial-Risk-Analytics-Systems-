# from fastapi import FastAPI
# from pydantic import BaseModel
# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
#
# # Initialize the FastAPI app
# app = FastAPI()
#
# # Define a request model to parse the input data
# class CreditRiskRequest(BaseModel):
#     LIMIT_BAL: float
#     SEX: int
#     EDUCATION: int
#     MARRIAGE: int
#     AGE: int
#     PAY_1: int
#     PAY_2: int
#     PAY_3: int
#     PAY_4: int
#     PAY_5: int
#     PAY_6: int
#     BILL_AMT1: float
#     BILL_AMT2: float
#     BILL_AMT3: float
#     BILL_AMT4: float
#     BILL_AMT5: float
#     BILL_AMT6: float
#     PAY_AMT1: float
#     PAY_AMT2: float
#     PAY_AMT3: float
#     PAY_AMT4: float
#     PAY_AMT5: float
#     PAY_AMT6: float
#
# # Load the pre-trained model
# model = xgb.XGBClassifier()
# model.load_model("best_xgb_model.json")
#
# # Define the feature columns used for training the model
# feature_columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
#                    'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
#                    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
#                    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
#
# # Define an API route for predictions
# @app.post("/predict/")
# async def predict(request: CreditRiskRequest):
#     # Convert the incoming data into a dataframe
#     input_data = pd.DataFrame([request.dict()])
#
#     input_data['Payment_Value'] = (input_data['PAY_AMT1'] + input_data['PAY_AMT2'] +
#                                    input_data['PAY_AMT3'] + input_data['PAY_AMT4'] +
#                                    input_data['PAY_AMT5'] + input_data['PAY_AMT6'])
#
#     input_data['Dues'] = ((input_data['BILL_AMT1'] + input_data['BILL_AMT2'] +
#                            input_data['BILL_AMT3'] + input_data['BILL_AMT4'] +
#                            input_data['BILL_AMT5']) -
#                           (input_data['PAY_AMT1'] + input_data['PAY_AMT2'] +
#                            input_data['PAY_AMT3'] + input_data['PAY_AMT4'] +
#                            input_data['PAY_AMT5'] + input_data['PAY_AMT6']))
#
#     # Calculate EAD
#     input_data['EAD'] = input_data['Dues'] - input_data['Payment_Value']
#
#     # Preprocess the data (scaling)
#     scaler = StandardScaler()
#     input_data_scaled = scaler.fit_transform(input_data[feature_columns])
#
#     # Predict using the pre-trained model
#     prediction = model.predict(input_data_scaled)
#
#     # Return the result including the calculated EAD
#     return {
#         "prediction": int(prediction[0]),
#         "ead": input_data['EAD'].iloc[0]  # Return the calculated EAD
#     }
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8006)
#
#

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Initialize the FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this list with your front-end URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Define a request model to parse the input data
class CreditRiskRequest(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_1: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


# Load the pre-trained model
model = xgb.XGBClassifier()
model.load_model("best_xgb_model.json")

# Define the feature columns used for training the model
feature_columns = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
                   'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                   'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']


# Define an API route for predictions
@app.post("/predict/")
async def predict(request: CreditRiskRequest):
    # Convert the incoming data into a dataframe
    input_data = pd.DataFrame([request.dict()])

    input_data['Payment_Value'] = (input_data['PAY_AMT1'] + input_data['PAY_AMT2'] +
                                   input_data['PAY_AMT3'] + input_data['PAY_AMT4'] +
                                   input_data['PAY_AMT5'] + input_data['PAY_AMT6'])

    input_data['Dues'] = ((input_data['BILL_AMT1'] + input_data['BILL_AMT2'] +
                           input_data['BILL_AMT3'] + input_data['BILL_AMT4'] +
                           input_data['BILL_AMT5']) -
                          (input_data['PAY_AMT1'] + input_data['PAY_AMT2'] +
                           input_data['PAY_AMT3'] + input_data['PAY_AMT4'] +
                           input_data['PAY_AMT5'] + input_data['PAY_AMT6']))

    # Calculate EAD
    input_data['EAD'] = input_data['Dues'] - input_data['Payment_Value']

    # Preprocess the data (scaling)
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data[feature_columns])

    # Predict using the pre-trained model
    prediction = model.predict(input_data_scaled)

    # Return the result including the calculated EAD
    return {
        "prediction": int(prediction[0]),
        "ead": input_data['EAD'].iloc[0]  # Return the calculated EAD
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8006)
