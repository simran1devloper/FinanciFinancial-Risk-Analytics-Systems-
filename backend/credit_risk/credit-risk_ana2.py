#credit-risk_ana2.py
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the dataset
credit_risk = pd.read_csv("./UCI_Credit_Card.csv")
credit_risk.drop('ID', axis=1, inplace=True)

# Prepare data
credit_risk["Defaulted"] = credit_risk["default.payment.next.month"]
credit_risk.drop("default.payment.next.month", axis=1, inplace=True)

# Feature engineering
credit_risk['Payment_Value'] = credit_risk['PAY_AMT1'] + credit_risk['PAY_AMT2'] + credit_risk['PAY_AMT3'] + credit_risk['PAY_AMT4'] + credit_risk['PAY_AMT5'] + credit_risk['PAY_AMT6']
credit_risk['Dues'] = (credit_risk['BILL_AMT1'] + credit_risk['BILL_AMT2'] + credit_risk['BILL_AMT3'] + credit_risk['BILL_AMT4'] + credit_risk['BILL_AMT5']) - (credit_risk['PAY_AMT1'] + credit_risk['PAY_AMT2'] + credit_risk['PAY_AMT3'] + credit_risk['PAY_AMT4'] + credit_risk['PAY_AMT5'] + credit_risk['PAY_AMT6'])

# Preprocessing
encoders_nums = {"SEX": {1: 0, 2: 1}}
credit_risk = credit_risk.replace(encoders_nums)
credit_risk.drop(['EDUCATION', 'MARRIAGE'], axis=1, inplace=True)

# Train/test split
y = credit_risk['Defaulted']
X = credit_risk.drop(['Defaulted'], axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Handling class imbalance using SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# Model training using XGBoost with GridSearchCV
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

xgb_clf = xgb.XGBClassifier()

param_grid = {
    'max_depth': [3, 10, 2],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Performing GridSearchCV to find the best model
grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, scoring='roc_auc', cv=3, verbose=0)
grid_search.fit(X_train, y_train)

# Get the best parameters and estimator
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best parameters found by GridSearchCV:", best_params)

# Evaluate the best model
train_class_preds_probs = best_model.predict_proba(X_train)[:, 1]
test_class_preds_probs = best_model.predict_proba(X_test)[:, 1]

from sklearn.metrics import accuracy_score, roc_auc_score

train_class_preds = [1 if prob >= 0.5 else 0 for prob in train_class_preds_probs]
test_class_preds = [1 if prob >= 0.5 else 0 for prob in test_class_preds_probs]

train_accuracy = accuracy_score(y_train, train_class_preds)
test_accuracy = accuracy_score(y_test, test_class_preds)

train_auc = roc_auc_score(y_train, train_class_preds_probs)
test_auc = roc_auc_score(y_test, test_class_preds_probs)

print("Train accuracy:", train_accuracy)
print("Test accuracy:", test_accuracy)
print("Train AUC:", train_auc)
print("Test AUC:", test_auc)

# Save the best model
best_model.save_model("best_xgb_model.json")
print("Best model saved as 'best_xgb_model.json'")
