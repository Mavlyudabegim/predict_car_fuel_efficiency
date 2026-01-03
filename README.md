# 🏎️ Car Fuel Efficiency Predictor
A mini data science project featuring a custom-trained **Regularized Linear Regression** model and an interactive web interface for real-time predictions.

 🔗 [Live Demo Link](https://predictcarfuelefficiency.streamlit.app/)

## 📌 Project Overview
The goal of this project is to predict a vehicle's fuel efficiency (MPG) based on its technical specifications. Unlike a simple "black box" model, 
this project demonstrates the full lifecycle of machine learning: from data cleaning and statistical validation in a Jupyter Notebook 
to model serialization and cloud deployment.

## 🛠️ Tech Stack
* **Language:** Python 3.13.9
* **Data Analysis:** Pandas, NumPy, Seaborn (for EDA)
* **Modeling:** Linear Regression with $L2$ Regularization (Ridge)
* **Web Framework:** Streamlit
* **Serialization:** Joblib

## 📊 The Model Workflow
This project follows a professional **Training vs. Inference** pipeline:
### 1. Training Pipeline (linear_regression.ipynb):
* **Preprocessing:** Handled missing values using mean/zero imputation based on feature distribution.
* **Feature Engineering:** Selected key drivers: Engine Displacement, Horsepower, Vehicle Weight, and Model Year.
* **Regularization:** Implemented Ridge Regression ($\lambda = 0.001$) to prevent overfitting and ensure the model generalizes well to new car models.
* **Validation:** Evaluated model performance using Root Mean Squared Error (RMSE) across different data splits.
* 
### 2. Deployment Pipeline (app.py):
* The trained model weights are serialized into a .pkl file.
* A Streamlit-based UI allows users to input their own car specs and receive an instant prediction via the loaded model parameters.

## 🚀 How to Run Locally
1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/car-fuel-predictor.git
cd car-fuel-predictor
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app:**
```bash
streamlit run app.py
```

## 📈 Key Insights
* **Weight vs. MPG:**
The model confirms a strong negative correlation; as vehicle weight increases, fuel efficiency significantly drops.

* **Model Year:**
Newer vehicles show a positive coefficient, reflecting the advancement in engine efficiency over time.

## 💡 Future Improvements
* Add categorical encoding for "Origin" (USA, Europe, Asia) to see if regional manufacturing impact efficiency.
* Implement a comparison tool to compare two different cars side-by-side.
