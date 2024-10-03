# Customers Default Payment Prediction

## Background Project

A bank aims to identify credit card customers who are at risk of defaulting on their payments using the available credit card payment history data. The data includes information on payment amounts, credit card bills, customer demographics, payment history, and late payment details. The bank seeks to improve its ability to predict customers who are likely to default on their payments using machine learning models.

## Objective

To implement machine learning classification algorithms, specifically Logistic Regression, SVM, and KNN, including understanding their concepts, preparing the data, and using appropriate hyperparameters.

## Tools

[<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />](https://pandas.pydata.org/)
[<img src="https://img.shields.io/badge/Seaborn-388E3C?style=for-the-badge&logo=seaborn&logoColor=white" alt="Seaborn" />](https://seaborn.pydata.org/)
[<img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="Numpy" />](https://numpy.org/)
[<img src="https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib" />](https://matplotlib.org/)
[<img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="SciPy" />](https://www.scipy.org/)
[<img src="https://img.shields.io/badge/Scikit%20learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn" />](https://scikit-learn.org/)
[<img src="https://img.shields.io/badge/XGBoost-016E54?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost" />](https://xgboost.ai/)

## Project Overview

### Problem

To create a machine learning classification model to predict customer default payment using the available data.

### Data Collection

The project utilizes the [Default of Credit Card Clients Data Set](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) from the UCI Machine Learning Repository. This dataset includes various features related to credit card usage and payment history.

### Data Processing and Modeling

The entire data processing and modeling process is consolidated into a single notebook, `Credit_Card_Default_Prediction.ipynb`. This notebook includes the following steps:

1. **Data Preprocessing**: Cleaning the data, handling missing values, and preparing the dataset for modeling.
2. **Modeling**: Implementing and training the following models:
    - **Logistic Regression**: A basic classification algorithm used to predict binary outcomes.
    - **K-Nearest Neighbors (KNN)**: An algorithm that classifies data based on the proximity to other data points in the feature space.
    - **Support Vector Machines (SVM)**: A model that finds the optimal hyperplane to separate different classes.
3. **Evaluation**: Evaluating model performance using metrics such as accuracy, precision, recall, and F1-score to assess how well the models predict customer defaults.

### Conclusion

Based on the data analysis, it is evident that customers who pay on time outnumber those who do not, indicating an imbalance in the data. All customers have the potential to default, but the risk is higher for those who have been late for more than two months. Demographically, unmarried and married customers have the highest default rates at 22.7% and 20.5%, respectively. In terms of education, customers with high school and bachelor's degrees have the highest default rates at 16.9% and 24.4%, respectively. Gender analysis shows that males have a slightly higher default risk (22.5%) compared to females (20.7%). These insights are crucial for credit risk management, emphasizing the need for special attention to high-risk groups.

Model evaluation reveals the following:
- **Logistic Regression**: Consistent F1-Score between training and test data, with stable performance and minimal variation from cross-validation, indicating no significant overfitting or underfitting and balanced performance with a weighted F1-score.
- **KNN**: Shows signs of overfitting as the F1-Score on training data is higher than on test data, though cross-validation indicates stable performance with small variations.
- **SVM**: Also shows slight overfitting with reduced performance on test data compared to training data, but maintains good consistency based on cross-validation.

After tuning, the **Logistic Regression** model showed significant performance improvement with a higher F1-score on training data (from 0.8057 to 0.8312) and test data (from 0.7854 to 0.8061). The changes in coefficients, especially for the eighth and ninth features, and the adjustment of the intercept indicate that the model has been better adjusted to the data distribution, providing more accurate predictions and enabling a deeper analysis of each feature's impact on the likelihood of default payment.

### Achievements

- **High Accuracy**: Achieved high accuracy in predicting default payments across different models.
- **Comprehensive Analysis**: Provided a thorough analysis of feature importance and model performance.
- **Enhanced Model Performance**: Significant improvement in model performance after tuning, particularly for Logistic Regression.

## Deployment

You can access the live model via the following link: [Credit Card Default Prediction on Hugging Face](https://huggingface.co/spaces/Gieorgie/Credit_Card_Default_Prediction).

