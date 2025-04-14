import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Define ensemble model class
class CustomEnsembleModel:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict(self, X):
        preds = [model.predict(X) for model in self.models]
        return np.average(preds, axis=0, weights=self.weights)

# Load saved components
model = joblib.load("best_model_ensemble.pkl")
template_df = joblib.load("template_input_df.pkl")
scaler = joblib.load("scaler.pkl")

# Set default neighborhood encoding
DEFAULT_NEIGHBORHOOD_ENC = 180000  # Fixed number used during training

# Page setup
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Predictor")
st.markdown("Enter house details to estimate the sale price.")

# Input UI
col1, col2 = st.columns(2)
with col1:
    OverallQuality = st.slider("Overall Quality (1–10)", 1, 10, 5)
    GarageCars = st.slider("Garage Capacity (cars)", 0, 4, 2)
    YearBuilt = st.number_input("Year Built", 1900, 2023, 2005)
with col2:
    GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 400, 6000, 1500)
    TotalBsmtSF = st.number_input("Basement Area (sq ft)", 0, 3000, 800)

# Create input row
input_df = template_df.copy()
input_df["OverallQuality"] = OverallQuality
input_df["GarageCars"] = GarageCars
input_df["YearBuilt"] = YearBuilt
input_df["GrLivArea"] = GrLivArea
input_df["TotalBsmtSF"] = TotalBsmtSF
input_df["OverallQual_GrLivArea"] = OverallQuality * GrLivArea
input_df["GarageCars_YearBuilt"] = GarageCars * YearBuilt
input_df["Qual_Bsmt"] = OverallQuality * TotalBsmtSF
input_df["Year_Overall"] = YearBuilt * OverallQuality
input_df["Neighborhood_enc"] = DEFAULT_NEIGHBORHOOD_ENC

# Ensure correct order for scaler
try:
    input_df = input_df[scaler.feature_names_in_]
except Exception as e:
    st.error(f"Input feature mismatch. Details: {e}")
    st.stop()

# Apply scaler
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

# Predict and display result
if st.button("Predict Price"):
    try:
        prediction_log = model.predict(input_scaled)[0]
        prediction = np.expm1(prediction_log)
        if np.isnan(prediction) or np.isinf(prediction):
            st.error("Prediction failed. Invalid output.")
        else:
            st.success(f"Estimated House Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Footer
st.caption("Model trained using Linear, Lasso, XGBoost, and LightGBM in a weighted ensemble.")
