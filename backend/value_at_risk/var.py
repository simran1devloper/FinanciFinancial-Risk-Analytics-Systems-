# from fastapi import FastAPI
# from pydantic import BaseModel
# import yfinance as yf
# import numpy as np
# from arch import arch_model
# import pandas as pd
#
# app = FastAPI()
#
# @app.get("/predict_var/")
# async def predict_var(ticker: str, start_date: str, end_date: str):
#     # Download the data
#     sp = yf.download(ticker, start=start_date, end=end_date)
#
#     # Ensure the dataframe has data
#     if sp.empty:
#         return {"error": "No data found for the given ticker and date range."}
#
#     # Process the data
#     sp['returns'] = sp['Adj Close'].pct_change().dropna()
#     sp = sp.replace([np.inf, -np.inf], np.nan).dropna()
#     if sp['returns'].isnull().all():
#         return {"error": "All data points are NaN after cleaning."}
#
#     # Fit the GARCH model
#     model = arch_model(sp['returns'] * 100, vol='GARCH', p=1, q=1)
#     results = model.fit(disp='off', show_warning=False)
#     forecasts = results.forecast(start="2023-01-01", reindex=False)
#     cond_mean = forecasts.mean
#     cond_var = forecasts.variance
#     q = model.distribution.ppf([0.01, 0.05, 0.1])
#
#     # Compute Value at Risk (VaR)
#     value_at_risk = -cond_mean.values - np.sqrt(cond_var).values * q[None:]
#     value_at_risk_df = pd.DataFrame(value_at_risk, columns=["1%", "5%", "10%"], index=cond_var.index)
#
#     # Return a compact summary
#     summary = value_at_risk_df.head(3).to_dict()  # Adjusted to return only the first 3 rows
#
#     return summary
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8008)
#

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import numpy as np
from arch import arch_model
import pandas as pd

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, or specify specific origins here
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


@app.get("/predict_var/")
async def predict_var(ticker: str, start_date: str, end_date: str):
    # Download the data
    sp = yf.download(ticker, start=start_date, end=end_date)

    # Ensure the dataframe has data
    if sp.empty:
        return {"error": "No data found for the given ticker and date range."}

    # Process the data
    sp['returns'] = sp['Adj Close'].pct_change().dropna()
    sp = sp.replace([np.inf, -np.inf], np.nan).dropna()
    if sp['returns'].isnull().all():
        return {"error": "All data points are NaN after cleaning."}

    # Fit the GARCH model
    model = arch_model(sp['returns'] * 100, vol='GARCH', p=1, q=1)
    results = model.fit(disp='off', show_warning=False)
    forecasts = results.forecast(start="2023-01-01", reindex=False)
    cond_mean = forecasts.mean
    cond_var = forecasts.variance
    q = model.distribution.ppf([0.01, 0.05, 0.1])

    # Compute Value at Risk (VaR)
    value_at_risk = -cond_mean.values - np.sqrt(cond_var).values * q[None:]
    value_at_risk_df = pd.DataFrame(value_at_risk, columns=["1%", "5%", "10%"], index=cond_var.index)

    # Return a compact summary
    summary = value_at_risk_df.head(3).to_dict()  # Adjusted to return only the first 3 rows

    return summary


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8008)
