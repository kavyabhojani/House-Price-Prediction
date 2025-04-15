import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Debuggable Ensemble Model
class CustomEnsembleModel:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights
        self.model_names = ["Linear", "Lasso", "XGBoost", "LightGBM"]

    def predict(self, X):
        raw_preds = [model.predict(X)[0] for model in self.models]
        weighted_log_pred = sum(w * p for w, p in zip(self.weights, raw_preds))
        weighted_log_pred = np.clip(weighted_log_pred, 5, 13)  # 🛡 Clip to avoid overflow
        final_pred = np.expm1(weighted_log_pred)
        debug_info = {
            name: f"${np.expm1(p):,.2f}" for name, p in zip(self.model_names, raw_preds)
        }
        return final_pred, debug_info


# Load saved components
model = joblib.load("best_model_ensemble.pkl")
template_df = joblib.load("template_input_df.pkl")
scaler = joblib.load("scaler.pkl")

# Page layout
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("House Price Predictor")
st.markdown("Enter house details to estimate the sale price.")

# Input sliders
col1, col2 = st.columns(2)
with col1:
    OverallQuality = st.slider("Overall Quality (1–10)", 1, 10, 5)
    GarageCars = st.slider("Garage Capacity (cars)", 0, 4, 2)
    YearBuilt = st.number_input("Year Built", 1900, 2023, 2005)
with col2:
    GrLivArea = st.number_input("Above Ground Living Area (sq ft)", 400, 6000, 1500)
    TotalBsmtSF = st.number_input("Basement Area (sq ft)", 0, 3000, 800)

# Populate input row using template
input_df = template_df.copy()
input_df["OverallQuality"] = OverallQuality
input_df["GarageCars"] = GarageCars
input_df["YearBuilt"] = YearBuilt
input_df["GrLivArea"] = GrLivArea
input_df["TotalBsmtSF"] = TotalBsmtSF

# Interaction features
input_df["OverallQual_GrLivArea"] = OverallQuality * GrLivArea
input_df["GarageCars_YearBuilt"] = GarageCars * YearBuilt
input_df["Qual_Bsmt"] = OverallQuality * TotalBsmtSF
input_df["Year_Overall"] = YearBuilt * OverallQuality
input_df["Neighborhood_enc"] = 0  # Default to mean

# Cap interaction terms to avoid extreme predictions
input_df["OverallQual_GrLivArea"] = np.minimum(input_df["OverallQual_GrLivArea"], 7 * 2000)
input_df["GarageCars_YearBuilt"] = np.minimum(input_df["GarageCars_YearBuilt"], 3 * 2000)
input_df["Qual_Bsmt"] = np.minimum(input_df["Qual_Bsmt"], 10 * 1500)
input_df["Year_Overall"] = np.minimum(input_df["Year_Overall"], 10 * 2000)

# Warn if inputs are unusually high
if GrLivArea > 3000 or TotalBsmtSF > 2000 or YearBuilt > 2015:
    st.warning("Some inputs are unusually high — prediction may be affected.")

# Ensure input column order
try:
    input_df = input_df[scaler.feature_names_in_]
except Exception as e:
    st.error(f"Input feature mismatch: {e}")
    st.stop()

# Scale features
input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

# Predict button
if st.button("Predict Price"):
    try:
        prediction, model_debug = model.predict(input_scaled)
        if np.isnan(prediction) or np.isinf(prediction):
            st.error("Prediction failed. Invalid output.")
        elif prediction > 1_000_000:
            st.warning(f"Estimated House Price: ${prediction:,.2f}")
        else:
            st.success(f"Estimated House Price: ${prediction:,.2f}")

        # Show per-model breakdown
        st.markdown("#### 🔍 Model-wise Breakdown:")
        for model_name, price in model_debug.items():
            st.markdown(f"- **{model_name}**: {price}")

    except Exception as e:
        st.error(f"Prediction error: {e}")

# Footer
st.caption("Model trained using Linear, Lasso, XGBoost, and LightGBM in a weighted ensemble.")
