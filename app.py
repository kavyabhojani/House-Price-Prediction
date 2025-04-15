import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Safe Ensemble Model
class CustomEnsembleModel:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict(self, X):
        preds = [model.predict(X)[0] for model in self.models]
        weighted_log_pred = sum(w * p for w, p in zip(self.weights, preds))

        # Hard cap the output to prevent overflow
        weighted_log_pred = np.clip(weighted_log_pred, 5, 13)  # 5~13 = $148K to ~$442K
        return np.expm1(weighted_log_pred)

# Load components
model = joblib.load("best_model_ensemble.pkl")
template_df = joblib.load("template_input_df.pkl")
scaler = joblib.load("scaler.pkl")

# Page layout
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Predictor")
st.markdown("Enter house details to estimate the sale price.")

# Inputs
col1, col2 = st.columns(2)
with col1:
    OverallQuality = st.slider("Overall Quality (1–10)", 1, 10, 5)
    GarageCars = st.slider("Garage Capacity (cars)", 0, 4, 2)
    YearBuilt = st.number_input("Year Built", 1900, 2023, 2005)
with col2:
    GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 400, 6000, 1500)
    TotalBsmtSF = st.number_input("Basement Area (sq ft)", 0, 3000, 800)

# Construct input
input_df = template_df.copy()
input_df["OverallQuality"] = OverallQuality
input_df["GarageCars"] = GarageCars
input_df["YearBuilt"] = YearBuilt
input_df["GrLivArea"] = GrLivArea
input_df["TotalBsmtSF"] = TotalBsmtSF
input_df["OverallQual_GrLivArea"] = min(OverallQuality * GrLivArea, 14000)
input_df["GarageCars_YearBuilt"] = min(GarageCars * YearBuilt, 6000)
input_df["Qual_Bsmt"] = min(OverallQuality * TotalBsmtSF, 15000)
input_df["Year_Overall"] = min(YearBuilt * OverallQuality, 20000)
input_df["Neighborhood_enc"] = 0

# Match order and scale
try:
    input_df = input_df[scaler.feature_names_in_]
except Exception as e:
    st.error(f"Input mismatch: {e}")
    st.stop()

input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

# Predict
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_scaled)
        if np.isnan(prediction) or np.isinf(prediction):
            st.error("Prediction failed.")
        elif prediction > 1_000_000:
            st.warning(f"Estimated House Price: ${prediction:,.2f}")
        else:
            st.success(f"Estimated House Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.caption("Model trained using Linear, Lasso, XGBoost, and LightGBM in a weighted ensemble.")
