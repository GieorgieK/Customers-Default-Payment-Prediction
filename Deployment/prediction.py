import streamlit as st
import pandas as pd
import os
import joblib
import pickle

# Navigate to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Path to the model file
model_path = os.path.join(parent_dir, "model_logreg.pkl")

# Load the model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

def run():
    st.title('Credit Card Customers Default Prediction')

    #membuat sub header
    st.subheader('Model Default Prediction')

    with st.form('form_credit_default'):
        sex = st.selectbox('Sex', [1, 2], index=0, key='sex_selectbox')
        education_level = st.number_input('Education Level', 0, 6, 1, key='edu_slider')
        marital_status = st.number_input('Marital Status', 0, 3, 1, key='marital_slider')
        age = st.number_input('Age', 21, 69, 30, key='age_slider')
        limit_balance = st.number_input('Limit Balance', 10000, 800000, 200000, key='limit_bal_slider')
        bill_amt_1 = st.number_input('Bill Amount 1', 0, 964511, 50000, key='bill_amt1_slider')
        bill_amt_2 = st.number_input('Bill Amount 2', 0, 983931, 60000, key='bill_amt2_slider')
        bill_amt_3 = st.number_input('Bill Amount 3', 0, 1664089, 70000, key='bill_amt3_slider')
        bill_amt_4 = st.number_input('Bill Amount 4', 0, 891586, 80000, key='bill_amt4_slider')
        bill_amt_5 = st.number_input('Bill Amount 5', 0, 927171, 90000, key='bill_amt5_slider')
        bill_amt_6 = st.number_input('Bill Amount 6', 0, 961664, 100000, key='bill_amt6_slider')
        pay_amt_1 = st.number_input('Pay Amount 1', 0, 873552, 10000, key='pay_amt1_slider')
        pay_amt_2 = st.number_input('Pay Amount 2', 0, 1684259, 20000, key='pay_amt2_slider')
        pay_amt_3 = st.number_input('Pay Amount 3', 0, 896040, 30000, key='pay_amt3_slider')
        pay_amt_4 = st.number_input('Pay Amount 4', 0, 621000, 40000, key='pay_amt4_slider')
        pay_amt_5 = st.number_input('Pay Amount 5', 0, 426529, 50000, key='pay_amt5_slider')
        pay_amt_6 = st.number_input('Pay Amount 6', 0, 528666, 60000, key='pay_amt6_slider')
        
        # Placeholder values for 'pay_0' to 'pay_6'
        pay_0 = 0
        pay_2 = 0
        pay_3 = 0
        pay_4 = 0
        pay_5 = 0
        pay_6 = 0

        submitted = st.form_submit_button('Predict')

    data = {
        'limit_balance': limit_balance,
        'sex': sex,
        'education_level': education_level,
        'marital_status': marital_status,
        'age': age,
        'pay_0': pay_0,
        'pay_2': pay_2,
        'pay_3': pay_3,
        'pay_4': pay_4,
        'pay_5': pay_5,
        'pay_6': pay_6,
        'bill_amt_1': bill_amt_1,
        'bill_amt_2': bill_amt_2,
        'bill_amt_3': bill_amt_3,
        'bill_amt_4': bill_amt_4,
        'bill_amt_5': bill_amt_5,
        'bill_amt_6': bill_amt_6,
        'pay_amt_1': pay_amt_1,
        'pay_amt_2': pay_amt_2,
        'pay_amt_3': pay_amt_3,
        'pay_amt_4': pay_amt_4,
        'pay_amt_5': pay_amt_5,
        'pay_amt_6': pay_amt_6,
    }
    
    features = pd.DataFrame(data, index=[0])
    st.write("## User Input Features")
    st.write(features)

    if submitted:
        prediction = model.predict(features)
        st.subheader('Prediction')
        st.write('Default Payment Next Month:', 'Yes' if prediction[0] == 1 else 'No')

if __name__ == '__main__':
    run()