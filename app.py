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
        preds = []
        for i, model in enumerate(self.models):
            try:
                raw_pred = model.predict(X)
                raw_pred = np.clip(raw_pred, a_min=0, a_max=18)  # avoid overflow
                final_pred = np.expm1(raw_pred)
                preds.append(final_pred)
            except Exception as e:
                print(f"Model {i} failed: {e}")
                preds.append(np.zeros(X.shape[0]))  # fallback
        weighted_preds = sum(w * p for w, p in zip(self.weights, preds))
        return weighted_preds

# Load the model and the expected training column order
model = joblib.load("best_model_ensemble.pkl")
model_columns = joblib.load("model_columns.pkl")

# Streamlit UI
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

# Interaction features
OverallQual_GrLivArea = OverallQuality * GrLivArea
GarageCars_YearBuilt = GarageCars * YearBuilt
Qual_Bsmt = OverallQuality * TotalBsmtSF
Year_Overall = YearBuilt * OverallQuality
Neighborhood_enc = 180000  # placeholder for encoded categorical value

# Construct input DataFrame
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

# Reindex to match model input columns
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Run prediction and display result
if st.button("Predict Price"):
    try:
        price = model.predict(input_df)
        if np.isinf(price[0]) or np.isnan(price[0]):
            st.error("Prediction resulted in an invalid value. Please adjust the inputs.")
        else:
            st.success(f"Estimated House Price: ${price[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Footer
st.caption("Model trained using XGBoost, Lasso, Linear, LightGBM — combined in a weighted ensemble.")
