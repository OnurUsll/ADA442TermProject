# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
import joblib



# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Bank Marketing Prediction App", layout="wide")
st.title("Bank Term Deposit Subscription Prediction")
st.markdown("""
This app predicts whether a client will subscribe to a bank term deposit.
Fill out the form below and click **Predict**.
""")

# --- CUSTOM TRANSFORMER (Must match training pipeline) ---
class CustomFeaturesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['previous_contact'] = (X['pdays'] != 999).astype(int)
        X['unemployed'] = X['job'].isin(['student', 'retired', 'unemployed']).astype(int)
        return X

# --- LOAD PIPELINE ---
try:
    with open('full_pipeline.pkl', 'rb') as f:
        model_pipeline = pickle.load(f)
except Exception as e:
    st.error("Model pipeline could not be loaded.")
    st.exception(e)
    st.stop()

# --- CENTERED FORM LAYOUT ---
left, center, right = st.columns([1, 2, 1])
with center:
    st.header("Client Information")

    # === INPUT FIELDS ===
    age = st.slider("Age", 18, 100, 30)
    job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                               'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'])
    marital = st.selectbox("Marital Status", ['single', 'married', 'divorced'])
    education = st.selectbox("Education", ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
                                           'professional.course', 'university.degree'])
    default = st.selectbox("Has Credit in Default?", ['no', 'yes'])
    housing = st.selectbox("Has Housing Loan?", ['no', 'yes'])
    loan = st.selectbox("Has Personal Loan?", ['no', 'yes'])
    contact = st.selectbox("Contact Communication Type", ['cellular', 'telephone'])
    month = st.selectbox("Month of Last Contact", ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                   'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = st.selectbox("Day of Week of Last Contact", ['mon', 'tue', 'wed', 'thu', 'fri'])
    duration = st.slider("Duration of Last Contact (seconds)", 0, 5000, 500)
    campaign = st.slider("Number of Contacts During this Campaign", 1, 50, 1)
    pdays = st.slider("Days Since Last Contact (999 = never contacted)", 0, 999, 999)
    previous = st.slider("Number of Contacts Before this Campaign", 0, 50, 0)
    poutcome = st.selectbox("Outcome of Previous Campaign", ['failure', 'nonexistent', 'success'])
    emp_var_rate = st.slider("Employment Variation Rate", -3.4, 1.4, 0.0)
    cons_price_idx = st.slider("Consumer Price Index", 92.0, 95.0, 93.5)
    cons_conf_idx = st.slider("Consumer Confidence Index", -50.0, -26.0, -40.0)
    euribor3m = st.slider("Euribor 3 Month Rate", 0.6, 5.0, 3.0)
    nr_employed = st.slider("Number of Employees (in thousands)", 4900, 5300, 5100)

    # === PREDICTION ===
    if st.button("Predict"):
        input_data = pd.DataFrame({
            'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
            'default': [default], 'housing': [housing], 'loan': [loan], 'contact': [contact],
            'month': [month], 'day_of_week': [day_of_week], 'duration': [duration],
            'campaign': [campaign], 'pdays': [pdays], 'previous': [previous], 'poutcome': [poutcome],
            'emp.var.rate': [emp_var_rate], 'cons.price.idx': [cons_price_idx],
            'cons.conf.idx': [cons_conf_idx], 'euribor3m': [euribor3m], 'nr.employed': [nr_employed]
        })

        try:
            prediction = model_pipeline.predict(input_data)
            prediction_proba = model_pipeline.predict_proba(input_data)
        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)
            st.stop()

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success("✅ The client is likely to subscribe to the term deposit!")
        else:
            st.error("❌ The client is unlikely to subscribe to the term deposit.")
        st.markdown(f"**Subscription Probability:** `{prediction_proba[0][1]:.2%}`")

        # === BAR CHART FOR PROBABILITY ===
        fig, ax = plt.subplots(figsize=(4, 0.5))
        ax.barh(0, prediction_proba[0][1], color='green')
        ax.barh(0, 1 - prediction_proba[0][1], left=prediction_proba[0][1], color='red')
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        st.pyplot(fig)

        # === FACTOR INSIGHTS ===
        st.subheader("Key Factors")
        st.write("""
        - **Duration**: Longer calls = more interest  
        - **Past outcome**: Previous success helps  
        - **Month & economic indicators** matter  
        """)
    else:
        st.subheader("Instructions")
        st.write("Fill in the form and click Predict.")

# --- FOOTER ---
st.markdown("---")
st.subheader("About")
st.info("This app was built for ADA 442 Statistical Learning course.")
st.markdown("**Team Members:** Abdullah Doğanay, Onur Uslu, Emirhan Yılmaz, Talha El Bah")
