import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Define feature engineering function - must be identical to the one used to create the pipeline
def feature_engineering(X):
    X_ = X.copy()
    # Add the engineered features we created earlier
    X_['total_contacts'] = X_['campaign'] + X_['previous']
    X_['euribor_emp_var'] = X_['euribor3m'] * X_['emp.var.rate']
    return X_

# Load the model pipeline
with open('full_pipeline.pkl', 'rb') as f:
    model_pipeline = pickle.load(f)

# Set page configuration
st.set_page_config(page_title="Bank Marketing Prediction App", layout="wide")

# Title
st.title("Bank Term Deposit Subscription Prediction")
st.markdown("""
This app predicts whether a client will subscribe to a bank term deposit based on various features.
Enter the client's details below and click on the 'Predict' button to get the prediction.
""")

# Create a sidebar for inputs
st.sidebar.header("Client Information")

# Define the input fields
# Age
age = st.sidebar.slider("Age", 18, 100, 30)

# Job type
job_options = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
              'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed']
job = st.sidebar.selectbox("Job", job_options)

# Marital status
marital_options = ['single', 'married', 'divorced']
marital = st.sidebar.selectbox("Marital Status", marital_options)

# Education
education_options = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
                     'professional.course', 'university.degree']
education = st.sidebar.selectbox("Education", education_options)

# Default
default = st.sidebar.selectbox("Has Credit in Default?", ['no', 'yes'])

# Housing loan
housing = st.sidebar.selectbox("Has Housing Loan?", ['no', 'yes'])

# Personal loan
loan = st.sidebar.selectbox("Has Personal Loan?", ['no', 'yes'])

# Contact
contact_options = ['cellular', 'telephone']
contact = st.sidebar.selectbox("Contact Communication Type", contact_options)

# Month
month_options = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month = st.sidebar.selectbox("Month of Last Contact", month_options)

# Day of week
day_options = ['mon', 'tue', 'wed', 'thu', 'fri']
day_of_week = st.sidebar.selectbox("Day of Week of Last Contact", day_options)

# Duration
duration = st.sidebar.slider("Duration of Last Contact (seconds)", 0, 5000, 500)

# Campaign
campaign = st.sidebar.slider("Number of Contacts Performed During this Campaign", 1, 50, 1)

# Pdays
pdays = st.sidebar.slider("Days Since Client was Last Contacted (999 = never contacted)", 0, 999, 999)

# Previous
previous = st.sidebar.slider("Number of Contacts Before this Campaign", 0, 50, 0)

# Poutcome
poutcome_options = ['failure', 'nonexistent', 'success']
poutcome = st.sidebar.selectbox("Outcome of Previous Marketing Campaign", poutcome_options)

# Economic indicators (these could be preset or allow user input)
emp_var_rate = st.sidebar.slider("Employment Variation Rate", -3.4, 1.4, 0.0)
cons_price_idx = st.sidebar.slider("Consumer Price Index", 92.0, 95.0, 93.5)
cons_conf_idx = st.sidebar.slider("Consumer Confidence Index", -50.0, -26.0, -40.0)
euribor3m = st.sidebar.slider("Euribor 3 Month Rate", 0.6, 5.0, 3.0)
nr_employed = st.sidebar.slider("Number of Employees (in thousands)", 4900, 5300, 5100)

# Button to make prediction
if st.sidebar.button("Predict"):
    # Create a dataframe with the input values
    input_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'day_of_week': [day_of_week],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome],
        'emp.var.rate': [emp_var_rate],
        'cons.price.idx': [cons_price_idx],
        'cons.conf.idx': [cons_conf_idx],
        'euribor3m': [euribor3m],
        'nr.employed': [nr_employed]
    })

    # Make prediction
    prediction = model_pipeline.predict(input_data)
    prediction_proba = model_pipeline.predict_proba(input_data)
    
    # Display result
    st.subheader("Prediction Results")
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Client Profile:")
        st.write(f"- **Age:** {age}")
        st.write(f"- **Job:** {job}")
        st.write(f"- **Education:** {education}")
        st.write(f"- **Contact:** {contact}")
        st.write(f"- **Duration of Call:** {duration} seconds")
        st.write(f"- **Previous Outcome:** {poutcome}")
    
    with col2:
        if prediction[0] == 1:
            st.success("The client is likely to subscribe to the term deposit! 📈")
        else:
            st.error("The client is unlikely to subscribe to the term deposit. 📉")
            
        # Display probability
        st.write(f"Probability of subscription: {prediction_proba[0][1]:.2%}")
        
        # Create a gauge chart for probability visualization
        fig, ax = plt.subplots(figsize=(4, 0.5))
        ax.barh(0, prediction_proba[0][1], color='green')
        ax.barh(0, 1-prediction_proba[0][1], left=prediction_proba[0][1], color='red')
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        st.pyplot(fig)
    
    # Explanation section
    st.subheader("Key Factors")
    st.write("""
    Some of the most important factors in predicting term deposit subscriptions include:
    - **Duration**: Longer calls typically indicate higher interest
    - **Previous outcome**: Past success is a good predictor of future success
    - **Month**: Certain months show higher subscription rates
    - **Age**: Different age groups have varying propensities to subscribe
    - **Economic indicators**: Market conditions significantly impact decisions
    """)

else:
    # Show sample data visualization when no prediction has been made yet
    st.subheader("Bank Marketing Campaign Analysis")
    st.write("""
    Welcome to the Bank Term Deposit Prediction App! 
    
    This tool helps predict whether a client will subscribe to a term deposit based on various factors including
    demographic information, contact details, and economic indicators.
    
    **How to use this tool:**
    1. Enter client information in the sidebar on the left
    2. Click the 'Predict' button to see the results
    3. Review the prediction and probability of subscription
    
    This model was trained on bank marketing campaign data and achieves good performance in predicting client responses.
    """)

# Add information about the project
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    "This app uses machine learning to predict if a client will subscribe to a term deposit based on various features. "
    "It was created as part of the ADA 442 Statistical Learning course project."
)
st.sidebar.markdown("Created by: Abdullah Doğanay, Onur Uslu, Emirhan Yılmaz, Talha El Bah")
