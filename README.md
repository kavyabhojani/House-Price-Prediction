# House-Price-Prediction

This project is a full machine learning pipeline that predicts house prices based on various property features. It includes data preprocessing, feature engineering, model building, evaluation, and deployment using Streamlit.

---

## Features
- End-to-end regression pipeline using Python
- Feature selection and interaction feature engineering
- Model comparison across:
  - Linear Regression
  - Lasso Regression
  - XGBoost
  - LightGBM
- Weighted Ensemble model for best performance
- Final app deployed with Streamlit

---

## Model Performance
| Model              | RMSE     | R² Score |
|---------------------|----------|------------|
| Linear Regression   | ~22,686  | 0.8879     |
| Lasso Regression    | ~22,739  | 0.8873     |
| XGBoost             | ~22,042  | 0.8941     |
| LightGBM            | ~22,016  | 0.8944     |
| **Ensemble Model**  | **21,280** | **0.9013** |

---

## Files
- `app.py`: Streamlit app code for deployment
- `best_model_ensemble.pkl`: Trained ensemble model
- `template_input_df.pkl`: Template input row used for safe predictions
- `scaler.pkl`: StandardScaler used to normalize inputs during training

---

## Steps Performed in Notebook

1. **Data Handling and Cleaning**
   - Loaded dataset and removed obvious outliers
   - Checked and imputed missing values where required

2. **Exploratory Data Analysis (EDA)**
   - Visualized top correlated features with `SalePrice`
   - Plotted scatter graphs to understand linear relationships

3. **Feature Engineering**
   - Created interaction features (e.g., `OverallQual * GrLivArea`)
   - Used target encoding for categorical variables (e.g., `Neighborhood`)

4. **Modeling**
   - Compared Linear, Lasso, XGBoost, and LightGBM regressors
   - Tuned hyperparameters using early stopping where applicable
   - Combined models using a weighted ensemble

5. **Evaluation**
   - Evaluated using RMSE and R²
   - Residual analysis to check model assumptions

6. **Deployment**
   - Saved best models and scalers using `joblib`
   - Created a simple Streamlit app with sliders and inputs
   - Ensured model inputs are safely scaled and aligned

---

## Usage Instructions
1. Clone the repository
2. Ensure all `.pkl` files are present
3. Run locally: `streamlit run app.py`

---

## Author
Kavya Bhojani  

---
