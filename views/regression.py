import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

def render():
    # --- Header ---
    st.markdown("""
        <div style="text-align:center; padding:25px; background:linear-gradient(135deg, #42a5f5, #7e57c2);
                    color:white; border-radius:10px; margin-bottom:20px;">
            <h2>💵 EMI Amount Predictor</h2>
            <p>Predict the maximum EMI amount a user can comfortably pay based on their financial profile.</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Model Selection ---
    model_files = {
        "Linear Regression": "D:/Percia_MTech/GUVI/python/Projects/emi_prediction/models/regression/Linear_Regression_pipeline.pkl",
        "Random Forest Regressor": "D:/Percia_MTech/GUVI/python/Projects/emi_prediction/models/regression/Random_Forest_pipeline.pkl",
        "XGBoost Regressor": "D:/Percia_MTech/GUVI/python/Projects/emi_prediction/models/regression/XGBoost_pipeline.pkl"
    }

    model_choice = st.selectbox("🔽 Choose a Regression Model", list(model_files.keys()))
    model_path = model_files[model_choice]

    # Load model
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        st.error(f"❌ Model file not found: {model_path}")
        return

    st.markdown("---")

    # --- User Inputs ---
    st.subheader("📥 Enter Loan & Applicant Details")

    col1, col2 = st.columns(2)

    with col1:
        income = st.number_input("💰 Monthly Income", min_value=10000, step=1000)
        loan_amount = st.number_input("🏦 Requested Loan Amount", min_value=100000, step=500)
        credit_score = st.slider("📊 Credit Score", 300, 900, 650)

    with col2:
        tenure = st.number_input("📆 Loan Tenure (months)", min_value=5, max_value=360, step=6)
        existing_loans = st.number_input("📉 Existing Loans", min_value=0, step=1)
        employment_years = st.number_input("💼 Employment Years", min_value=0, max_value=40, step=1)

    if st.button("🔮 Predict EMI"):
        # --- Build Input Data ---
        input_dict = {
            "income": income,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
            "tenure": tenure,
            "existing_loans": existing_loans,
            "employment_years": employment_years
        }
        input_data = pd.DataFrame([input_dict])

        # --- Align with expected features ---
        try:
            if hasattr(model, "feature_names_in_"):
                expected_features = model.feature_names_in_
            elif hasattr(model, "named_steps"):
                last_step = list(model.named_steps.values())[-1]
                expected_features = getattr(last_step, "feature_names_in_", input_data.columns)
            else:
                expected_features = input_data.columns

            aligned_data = pd.DataFrame(columns=expected_features)
            aligned_data.loc[0] = 0
            for col in input_data.columns:
                if col in aligned_data.columns:
                    aligned_data.at[0, col] = input_data[col].iloc[0]

            # --- Make Prediction ---
            prediction = model.predict(aligned_data)[0]

            st.success(f"💵 Predicted EMI Amount: ₹{prediction:,.2f}")

        except Exception as e:
            st.error(f"⚠️ Feature alignment failed: {e}")