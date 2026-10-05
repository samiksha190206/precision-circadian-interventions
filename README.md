# Precision Circadian Interventions

## Project Description

Precision Circadian Interventions is a machine-learning-based web application designed to analyze daily behavioral patterns related to sleep, physical activity, social media usage, and reported well-being.

The application provides:
- User registration and login
- Daily behavioral check-in
- K-Means behavioral clustering
- Random Forest prediction
- Personalized wellness recommendations
- Progress tracking and visualizations

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Plotly
- SQLite

## Machine Learning Models

- K-Means Clustering
- Logistic Regression
- Random Forest
- Gradient Boosting

## How to Run

1. Install Python.
2. Open this folder in VS Code or a terminal.
3. Install the required packages:

   pip install -r requirements.txt

4. Run the Streamlit application:

   streamlit run app.py

5. Open the local URL provided by Streamlit in a web browser.

## Source Files

app.py - Main Streamlit application

feature_columns.pkl - Feature column information

kmeans_model.pkl - Trained K-Means model

scaler.pkl - Feature scaling model

logistic_model.pkl - Trained Logistic Regression model

random_forest_model.pkl - Trained Random Forest model

gradient_boosting_model.pkl - Trained Gradient Boosting model

requirements.txt - Required Python libraries