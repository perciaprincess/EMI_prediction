def render():
    import streamlit as st
    import pandas as pd
    import joblib
    import os
    from pathlib import Path

    # --- Configuration: single default XGBoost model and optional label encoder ---
    MODEL_PATH = "../models/classification/EMI_Classification_Model_XGB.pkl"
    ENCODER_PATH = "models/classification/label_encoder.pkl"
    PREPROCESSOR_PATH = "models/classification/preprocessor.pkl"
    FEATURE_ORDER_PATH = "models/classification/feature_order.json"

    # --- Header ---
    st.markdown("""
        <div style="text-align:center; padding:20px; background:linear-gradient(135deg, #42a5f5, #7e57c2);
                    color:white; border-radius:10px; margin-bottom:16px;">
            <h2>🎯 EMI Eligibility Predictor</h2>
            <p>Predict Eligible, Not Eligible, or High Risk using the default XGBoost model.</p>
        </div>
    """, unsafe_allow_html=True)

    # --- Load model and optional artifacts ---
    if not Path(MODEL_PATH).is_file():
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        return
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as ex:
        st.error(f"❌ Failed to load model: {ex}")
        return

    preprocessor = joblib.load(PREPROCESSOR_PATH) if Path(PREPROCESSOR_PATH).is_file() else None
    label_encoder = joblib.load(ENCODER_PATH) if Path(ENCODER_PATH).is_file() else None
    feature_order = None
    if Path(FEATURE_ORDER_PATH).is_file():
        import json
        with open(FEATURE_ORDER_PATH, "r") as f:
            feature_order = json.load(f)

    st.markdown("---")
    st.markdown("### Enter Applicant Details")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("🎂 Age", min_value=18, max_value=70, value=30, key="age")
        monthly_salary = st.number_input("💰 Monthly Salary", min_value=0, step=1000, value=50000, key="monthly_salary")
        employment_type = st.selectbox("💼 Employment Type", ["Private", "Government", "Self-Employed"], key="employment_type")
        company_type = st.selectbox("🏢 Company Type", ["Startup", "Mid-Size", "Mnc", "Small"], key="company_type")
        marital_status = st.selectbox("💍 Marital Status", ["Single", "Married"], key="marital_status")
        education = st.selectbox("🎓 Education Level", ["High School", "Graduate", "Post Graduate", "Professional"], key="education")

    with col2:
        credit_score = st.slider("📊 Credit Score", 300, 900, 650, key="credit_score")
        existing_loans = st.number_input("📉 Existing Loans (count)", min_value=0, step=1, value=0, key="existing_loans")
        tenure = st.number_input("📆 Requested Tenure (months)", min_value=6, max_value=360, step=6, value=60, key="requested_tenure")
        emi_scenario = st.selectbox("📋 EMI Scenario", ["Home EMI", "Vehicle EMI", "Education EMI", "Personal Loan EMI"], key="emi_scenario")
        house_type = st.selectbox("🏠 House Type", ["Own", "Rented"], key="house_type")
        gender = st.selectbox("👤 Gender", ["Male", "Female"], key="gender")

    # Additional numeric fields (optional extras)
    st.markdown("#### Financial details (optional)")
    col3, col4 = st.columns(2)
    with col3:
        monthly_rent = st.number_input("Monthly Rent", min_value=0, value=5000, step=100, key="monthly_rent")
        current_emi_amount = st.number_input("Current EMI Amount (total)", min_value=0, value=0, step=100, key="current_emi_amount")
        requested_amount = st.number_input("Requested Amount", min_value=0, value=200000, step=1000, key="requested_amount")
    with col4:
        groceries_utilities = st.number_input("Groceries & Utilities", min_value=0, value=5000, step=100, key="groceries_utilities")
        other_monthly_expenses = st.number_input("Other Monthly Expenses", min_value=0, value=2000, step=100, key="other_monthly_expenses")
        emergency_fund = st.number_input("Emergency Fund", min_value=0, value=30000, step=500, key="emergency_fund")
        bank_balance = st.number_input("Bank Balance", min_value=0, value=20000, step=500, key="bank_balance")

    submitted = st.button("Submit Application")
    
    if submitted:
        # Build raw input dict (keys should match training raw feature names)
        raw = {
            "age": age,
            "monthly_salary": monthly_salary,
            "years_of_employment": 0,           # default if not asked; adjust if needed
            "monthly_rent": monthly_rent,
            "family_size": 1,
            "dependents": existing_loans,       # map if needed
            "school_fees": 0,
            "college_fees": 0,
            "travel_expenses": 0,
            "groceries_utilities": groceries_utilities,
            "other_monthly_expenses": other_monthly_expenses,
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "credit_score": credit_score,
            "bank_balance": bank_balance,
            "emergency_fund": emergency_fund,
            "requested_amount": requested_amount,
            "requested_tenure": tenure,
            "emi_scenario": emi_scenario,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "employment_type": employment_type,
            "company_type": company_type,
            "house_type": house_type,
            "emi_eligibility": 0,
            "max_monthly_emi": 0
        }
        input_df = pd.DataFrame([raw])

        # Compute engineered features exactly as done in training
        def compute_engineered(df):
            df = df.copy()
            df["total_expenses"] = (
                df["monthly_rent"] + df["school_fees"] + df["college_fees"] +
                df["travel_expenses"] + df["groceries_utilities"] +
                df["other_monthly_expenses"] + df["current_emi_amount"]
            )
            df["net_monthly_income"] = df["monthly_salary"] - df["total_expenses"]
            df["expense_to_income_ratio"] = df["total_expenses"] / (df["monthly_salary"] + 1e-9)
            df["dependency_ratio"] = df["dependents"] / (df["family_size"] + 1e-9)
            df["risk_score"] = ((1 - (df["credit_score"] - 300) / 600) * 0.7 +
                                (df["current_emi_amount"] / (df["monthly_salary"] + 1e-9)) * 0.3).clip(0,1)
            return df

        input_df = compute_engineered(input_df)

        # Align features for model input
        try:
            if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
                # model is a pipeline that will preprocess raw columns
                X_for_model = input_df
            elif preprocessor is not None:
                # apply external preprocessor then ensure feature order if available
                X_trans = preprocessor.transform(input_df)
                if feature_order:
                    X_for_model = pd.DataFrame(X_trans, columns=feature_order)
                else:
                    X_for_model = pd.DataFrame(X_trans)
            elif feature_order:
                # manual one-hot + align to feature_order
                d = pd.get_dummies(input_df, columns=[c for c in input_df.select_dtypes(include=["object"]).columns])
                for feat in feature_order:
                    if feat not in d.columns:
                        d[feat] = 0
                X_for_model = d[feature_order].astype(float)
            else:
                # fallback: pass numeric columns and label-encode categoricals
                df_tmp = input_df.copy()
                for c in df_tmp.select_dtypes(include=["object"]).columns:
                    df_tmp[c] = df_tmp[c].astype("category").cat.codes
                X_for_model = df_tmp

            # Predict
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_for_model)[:, 1][0]
            else:
                proba = None
            raw_pred = model.predict(X_for_model)[0]
            decoded = raw_pred
            if label_encoder is not None:
                try:
                    decoded = label_encoder.inverse_transform([int(raw_pred)])[0]
                except Exception:
                    decoded = raw_pred

            # Map and display result
            if isinstance(decoded, str):
                lab = decoded.lower()
                if "eligible" in lab and "not" not in lab:
                    st.success(f"✅ Eligible — {decoded}")
                elif "high" in lab:
                    st.warning(f"⚠️ High Risk — {decoded}")
                else:
                    st.error(f"❌ Not Eligible — {decoded}")
            else:
                # numeric-coded label mapping convention may vary; adjust as needed
                if int(raw_pred) == 0:
                    st.success("✅ Eligible")
                elif int(raw_pred) == 1:
                    st.warning("⚠️ High Risk")
                else:
                    st.error("❌ Not Eligible")

            if proba is not None:
                st.info(f"Approval probability: {proba:.2f}")

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
