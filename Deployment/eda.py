import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os



def run():
    # membuat judul
    st.title('Credit Card Customers Default Prediction')

    #membuat sub header
    st.subheader('EDA untuk Analysis Dataset Credit Card')

    #menambahkan deskripsi
    st.write('Page ini dibuat oleh Gieorgie')

    #mmebuat batas dengan garis lurus
    st.markdown('---')

    #show dataframe
    csv_file = os.path.join(os.path.dirname(__file__), "..", "P1G5_Set_1_gieorgie.csv")
    df = pd.read_csv(csv_file)
    st.dataframe(df)

    st.markdown('---')

    # List of numeric and categorical columns
    list_numeric = ['limit_balance', 'age', 'bill_amt_1', 'bill_amt_2', 'bill_amt_3', 'bill_amt_4', 'bill_amt_5', 'bill_amt_6', 'pay_amt_1', 'pay_amt_2', 'pay_amt_3', 'pay_amt_4', 'pay_amt_5', 'pay_amt_6']
    cat_columns = ['sex', 'education_level', 'marital_status', 'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']

    # Sidebar for user input
    st.sidebar.title("Choose Visualization")

    # Dropdown for selecting column type and column name
    selected_column_type = st.sidebar.radio("Select Column Type", ["Numeric", "Categorical"])

    if selected_column_type == "Numeric":
        selected_column = st.sidebar.selectbox("Select Numeric Column", list_numeric)

        # Display histogram based on user selection
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.hist(df[selected_column], bins=100)
        plt.title(f'Histogram of {selected_column}')
        st.pyplot(fig)

    elif selected_column_type == "Categorical":
        selected_column = st.sidebar.selectbox("Select Categorical Column", cat_columns)

        # Count the values of selected column
        value_counts = df[selected_column].value_counts()

        # Plotting based on user selection
        fig = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values)])
        fig.update_layout(title=f'Distribution of {selected_column}', xaxis_title='Categories', yaxis_title='Count')
        st.plotly_chart(fig)

if __name__ == '__main__':
    run()