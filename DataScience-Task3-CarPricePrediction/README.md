# 🚗 Car Price Prediction

## 📌 Oasis Infobyte Data Science Internship - Task 3

### 📖 Project Overview

This project builds a professional machine learning pipeline to predict the selling price of used cars using regression models. The workflow includes data cleaning, categorical value standardization, feature engineering, exploratory data analysis, model training, evaluation, and model saving.

---

## 🎯 Objective

The main objective is to predict the selling price of a car accurately by comparing multiple regression algorithms and selecting the best-performing model.

---

## 📂 Dataset

The project uses the publicly available car price dataset stored in the dataset folder.

### Key Features Used

- Car_Name
- Year
- Present_Price
- Kms_Driven
- Fuel_Type
- Seller_Type
- Transmission
- Owner
- Selling_Price

### Feature Engineering Added

- Brand extracted from Car_Name
- Car_Age derived from Year

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## 📊 EDA and Preprocessing

The notebook and script perform the following:

- Missing value and duplicate handling
- Standardization of categorical values
- Brand extraction from car names
- Selling price distribution analysis
- Fuel-type box plot analysis
- Price vs. year scatter plot
- Correlation heatmap
- One-hot encoding of categorical variables

---

## 🤖 Machine Learning Models

The following regression models are trained and compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

---

## 📈 Evaluation Metrics

The models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

A feature importance chart is also generated for the best-performing model.

---

## 🏆 Output Files

The project generates plots and a trained model in the outputs folder, including:

- selling_price_distribution.png
- price_vs_fuel_type.png
- price_vs_year.png
- correlation_heatmap.png
- feature_importance.png
- actual_vs_predicted.png
- best_car_price_model.pkl

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Open the Jupyter Notebook or run the Python script:

```bash
python DataScience-Task3-CarPricePrediction.py
```

---

## 👨‍💻 Author

Shaikh Danish Shaikh Umar Farooq

Oasis Infobyte Data Science Internship