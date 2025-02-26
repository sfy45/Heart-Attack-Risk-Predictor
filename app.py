import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model

# Prevent interactive Matplotlib issues
plt.switch_backend("Agg")


# Load trained model
model = load_model("heart_attack_risk_model")

# Streamlit App UI
st.title("🩺 Heart Attack Risk Prediction")

# Sidebar for navigation
st.sidebar.header("Upload Dataset or Enter Values Manually")

# Upload CSV option
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Dataset Preview:")
    st.write(df.head())

    # Predict using uploaded dataset
    if st.sidebar.button("Predict for Uploaded Dataset"):
        predictions = predict_model(model, data=df)
        st.write("📌 Predictions with Probability:")
        st.write(predictions[["prediction_label", "prediction_score"]])  # ✅ Show probability

# Manual input option
st.sidebar.subheader("Enter Data Manually")

def user_input_features():
    age = st.sidebar.slider("Age", 20, 100, 55)
    sex = st.sidebar.selectbox("Sex", [0, 1])
    cp = st.sidebar.slider("Chest Pain Type (CP)", 0, 3, 2)
    trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 130)
    chol = st.sidebar.slider("Cholesterol", 100, 400, 250)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.sidebar.slider("Resting ECG Results", 0, 2, 1)
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.sidebar.slider("Slope of ST Segment", 0, 2, 2)
    ca = st.sidebar.slider("Major Vessels Colored", 0, 4, 0)
    thal = st.sidebar.slider("Thalassemia", 0, 3, 3)

    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    return pd.DataFrame([data])

user_data = user_input_features()

# Predict for manual input
if st.sidebar.button("Predict for Manual Input"):
    prediction = predict_model(model, data=user_data)

    #  Get predicted class and probability
    predicted_label = prediction["prediction_label"][0]
    predicted_prob = prediction["prediction_score"][0]

    #  Custom decision threshold (adjust if needed)
    threshold = 0.6  # Default: 60% confidence for "High Risk"

    st.subheader("🏥 Prediction Result:")
    if predicted_prob > threshold:
        st.error(f"🚨 High Risk of Heart Attack! (Probability: {predicted_prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Heart Attack! (Probability: {predicted_prob:.2f})")

