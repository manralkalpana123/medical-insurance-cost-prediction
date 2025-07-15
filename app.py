import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from utils import preprocess_data

# Load model (you can use joblib or mlflow if deployed)
@st.cache_resource
def load_model():
    import mlflow.sklearn
    model_uri = "runs:/<YOUR_RUN_ID>/insurance_model"  # Update with your MLflow run_id
    model = mlflow.sklearn.load_model(model_uri)
    return model

st.title("🏥 Medical Insurance Cost Prediction")

age = st.slider("Age", 18, 64, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.slider("Children", 0, 5, 1)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "southeast", "northwest", "southwest"])

if st.button("Predict"):
    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    processed = preprocess_data(input_df)
    model = load_model()
    prediction = model.predict(processed)
    st.success(f"💰 Predicted Medical Cost: ${prediction[0]:.2f}")