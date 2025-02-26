# 🩺 Heart Attack Risk Prediction

This project is a **Machine Learning-based Heart Attack Risk Prediction System** built using **Streamlit** and **PyCaret**. It allows users to predict heart attack risks by either **uploading a CSV dataset** or **manually entering patient details**.

## 📌 Features
- Upload a **CSV file** containing patient data to get risk predictions.
- Manually input health parameters for **single patient prediction**.
- Displays **prediction probability** for better risk assessment.
- **User-friendly GUI** using Streamlit.

## 🛠️ Technologies Used
- **Python**
- **Streamlit** (for GUI)
- **PyCaret** (for Machine Learning)
- **Pandas** (for data handling)
- **Matplotlib** (for potential visualizations)

## 📊 Datasets Used
The model has been trained and tested using the following datasets from **Kaggle**:
1. **Heart Disease UCI Dataset** - [Link](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
2. **Framingham Heart Study Dataset** - [Link](https://www.kaggle.com/datasets/captainozlem/framingham-heart-study-dataset)

These datasets contain essential health parameters such as age, cholesterol levels, blood pressure, and lifestyle factors, which are used for training the model.

## 🚀 Installation & Setup

### 🔹 1. Clone the Repository
```bash
git clone https://github.com/sfy45/heart-attack-risk-prediction.git
cd heart-attack-risk-prediction
```

### 🔹 2. Install Dependencies
Make sure you have **Python 3.8+** installed. Then, run:
```bash
pip install -r requirements.txt
```

### 🔹 3. Run the Application
```bash
streamlit run app.py
```

## 🎯 Usage Guide

### 🏥 **Method 1: Upload a Dataset**
1. Click **"Upload CSV File"** in the sidebar.
2. Select a CSV file with patient health data.
3. Click **"Predict for Uploaded Dataset"** to get predictions.

### ✍️ **Method 2: Manual Input**
1. Use the sidebar sliders & dropdowns to input patient details.
2. Click **"Predict for Manual Input"** to check risk probability.

## 🤝 Contributing
**If you have a contribution to make, feel free to submit issues or pull requests. PRs are more than welcome!**

## ⚡ Contact
For any issues or suggestions, feel free to reach out:
📧 sophiasad1421@gmail.com

## ⭐ Show Some Support!
If you like this project, **give it a star ⭐ on GitHub!**

