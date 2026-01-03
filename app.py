import streamlit as st
import numpy as np
import os
import joblib

@st.cache_resource 
def load_model():
    if os.path.exists('fuel_model.pkl'):
        return joblib.load('fuel_model.pkl')
    else:
        return None

model_data = load_model()

if model_data is None:
    st.error("Model file 'fuel_model.pkl' not found. Please run your Jupyter Notebook first!")
    st.stop()

w_0 = model_data['w_0']
w = model_data['w']
features = model_data['features']

st.set_page_config(page_title="Fuel Efficiency Predictor", page_icon="🚗")

def predict_mpg(disp, hp, weight, year):
    X = np.array([disp, hp, weight, year])
    prediction = w_0 + X.dot(w)
    return prediction

st.title("🚗 Car Fuel Efficiency Predictor")
st.markdown("""
This app predicts the **Miles Per Gallon (MPG)** of a vehicle based on its technical specifications.
Built using a *Regularized Linear Regression* model.
""")
st.write(f"Model loaded with features: {', '.join(features)}")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        disp = st.number_input("Engine Displacement", value=170)
        hp = st.number_input("Horsepower", value=150)
        
    with col2:
        weight = st.number_input("Vehicle Weight (lbs)", value=3000)
        year = st.slider("Model Year", 1970, 2025, 2010)
    
    submit = st.form_submit_button("Predict MPG")

if submit:
    input_data = np.array([disp, hp, weight, year])
    
    prediction = w_0 + input_data.dot(w)
    
    st.metric("Predicted MPG", f"{prediction:.2f}")

    with st.expander("See the Math"):
        st.write(f"Formula: $y = {w_0:.2f} + ({w[0]:.4f} \\times Disp) + ({w[1]:.4f} \\times HP) + ({w[2]:.4f} \\times Weight) + ({w[3]:.4f} \\times Year)$")