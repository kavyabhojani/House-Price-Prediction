import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Define CustomEnsembleModel class (needed for joblib to load correctly)
class CustomEnsembleModel:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict(self, X):
        preds = [np.expm1(model.predict(X)) for model in self.models]
        weighted_preds = sum(w * p for w, p in zip(self.weights, preds))
        return weighted_preds

# Load the saved model and expected feature columns
model = joblib.load("best_model_ensemble.pkl")
model_columns = joblib.load("model_columns.pkl")

# Streamlit app UI
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Predictor")
st.markdown("Enter house details to get the estimated sale price.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    OverallQuality = st.slider("Overall Quality (1–10)", 1, 10, 5)
    GarageCars = st.slider("Garage Capacity (cars)", 0, 4, 2)
    YearBuilt = st.number_input("Year Built", 1900, 2023, 2005)

with col2:
    GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 400, 6000, 1500)
    TotalBsmtSF = st.number_input("Basement Area (sq ft)", 0, 3000, 800)

# Generate interaction features
OverallQual_GrLivArea = OverallQuality * GrLivArea
GarageCars_YearBuilt = GarageCars * YearBuilt
Qual_Bsmt = OverallQuality * TotalBsmtSF
Year_Overall = YearBuilt * OverallQuality
Neighborhood_enc = 180000  # Placeholder target encoding

# Create the input DataFrame
input_df = pd.DataFrame([{
    "OverallQuality": OverallQuality,
    "GrLivArea": GrLivArea,
    "GarageCars": GarageCars,
    "TotalBsmtSF": TotalBsmtSF,
    "YearBuilt": YearBuilt,
    "OverallQual_GrLivArea": OverallQual_GrLivArea,
    "GarageCars_YearBuilt": GarageCars_YearBuilt,
    "Qual_Bsmt": Qual_Bsmt,
    "Year_Overall": Year_Overall,
    "Neighborhood_enc": Neighborhood_enc
}])

#Reindex to match training feature structure
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Predict and show result
if st.button("Predict Price"):
    price = model.predict(input_df)
    st.success(f" Estimated House Price: **${price[0]:,.2f}**")

# Footer
st.caption("Model trained using XGBoost, Lasso, Linear, LightGBM — combined in a weighted ensemble.")
