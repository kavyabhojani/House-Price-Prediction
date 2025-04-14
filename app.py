import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Define CustomEnsembleModel class (for joblib compatibility)
class CustomEnsembleModel:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict(self, X):
        preds = []
        for i, model in enumerate(self.models):
            try:
                raw_pred = model.predict(X)
                raw_pred = np.clip(raw_pred, a_min=0, a_max=18)  # Prevent overflow
                final_pred = np.expm1(raw_pred)
                preds.append(final_pred)
            except Exception as e:
                print(f"Model {i} failed: {e}")
                preds.append(np.zeros(X.shape[0]))  # Fallback
        weighted_preds = sum(w * p for w, p in zip(self.weights, preds))
        return weighted_preds

# Load saved components
model = joblib.load("best_model_ensemble.pkl")
template_df = joblib.load("template_input_df.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit app UI
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Predictor")
st.markdown("Enter house details to estimate the sale price.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    OverallQuality = st.slider("Overall Quality (1–10)", 1, 10, 5)
    GarageCars = st.slider("Garage Capacity (cars)", 0, 4, 2)
    YearBuilt = st.number_input("Year Built", 1900, 2023, 2005)

with col2:
    GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 400, 6000, 1500)
    TotalBsmtSF = st.number_input("Basement Area (sq ft)", 0, 3000, 800)

# Use template to get all required columns, then override updated values
input_df = template_df.copy()
input_df["OverallQuality"] = OverallQuality
input_df["GrLivArea"] = GrLivArea
input_df["GarageCars"] = GarageCars
input_df["TotalBsmtSF"] = TotalBsmtSF
input_df["YearBuilt"] = YearBuilt
input_df["OverallQual_GrLivArea"] = OverallQuality * GrLivArea
input_df["GarageCars_YearBuilt"] = GarageCars * YearBuilt
input_df["Qual_Bsmt"] = OverallQuality * TotalBsmtSF
input_df["Year_Overall"] = YearBuilt * OverallQuality
input_df["Neighborhood_enc"] = 180000  # Placeholder for encoding

# Apply same scaling used in training
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

# Predict and display result
if st.button("Predict Price"):
    try:
        price = model.predict(input_scaled)
        if np.isinf(price[0]) or np.isnan(price[0]):
            st.error("Prediction failed: Invalid result. Please change input.")
        else:
            st.success(f"Estimated House Price: ${price[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Footer
st.caption("Model trained using XGBoost, Lasso, Linear, and LightGBM combined in a weighted ensemble.")
