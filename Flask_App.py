import joblib
import numpy as np
import pandas as pd
from routes import home, predict
from flask import Flask

app = Flask(__name__)

model = joblib.load('loan_model.pkl')

home()
predict(model)

Features = [
    'no_of_dependents',
    'income_annum',
    'loan_amount',
    'loan_term',
    'cibil_score',
    'residential_assets_value',
    'commercial_assets_value',
    'luxury_assets_value',
    'bank_asset_value',
    'education_ not graduate',
    'self_employed_ yes']

if __name__ == '__main__':
    app.run(debug=True)
