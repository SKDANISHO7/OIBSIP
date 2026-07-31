# 🚗 Car Price Prediction

## 📌 Oasis Infobyte Data Science Internship - Task 3

### 📖 Project Overview

The Car Price Prediction project is a Machine Learning regression project that predicts the selling price of a used vehicle based on its features such as manufacturing year, present price, fuel type, transmission, seller type, kilometers driven, and ownership history.

The project focuses on building multiple regression models, comparing their performance, and selecting the best-performing model for accurate price prediction.

---

## 🎯 Objective

The objective of this project is to build and compare multiple Machine Learning regression models to accurately predict the selling price of used vehicles.

---

## 📂 Dataset

The dataset contains information about used vehicles and their selling prices.

- Total Samples: 301
- Features: 8 (after preprocessing)

### Features

- Present Price
- Kms Driven
- Fuel Type
- Seller Type
- Transmission
- Owner
- Car Age
- Selling Price (Target)

### Target

- Selling Price

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

## 📊 Exploratory Data Analysis

The following analyses were performed:

- Dataset Overview
- Data Quality Assessment
- Missing Value Check
- Duplicate Value Check
- Statistical Summary
- Selling Price Distribution
- Correlation Heatmap
- Scatter Plot Analysis
- Categorical Feature Analysis
- Outlier Detection using Boxplots

---

## ⚙️ Feature Engineering

The following preprocessing steps were performed:

- Created **Car_Age** feature from manufacturing year.
- Removed unnecessary columns.
- One-Hot Encoding of categorical variables.
- Split dataset into training and testing sets.

---

## 🤖 Machine Learning Models

The following regression models were trained:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

---

## 📈 Model Evaluation

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 🏆 Best Model

The best-performing model was selected based on the highest **R² Score** and lowest prediction error.

The trained model was saved as:

```
best_car_price_model.pkl
```

---

## 📁 Project Structure

```
DataScience-Task3-CarPricePrediction/
│
├── dataset/
│   └── car_data.csv
│
├── outputs/
│   ├── actual_vs_predicted.png
│   ├── correlation_heatmap.png
│   ├── fuel_type_distribution.png
│   ├── kms_driven_boxplot.png
│   ├── kms_driven_vs_selling_price.png
│   ├── model_comparison.png
│   ├── owner_distribution.png
│   ├── present_price_boxplot.png
│   ├── present_price_vs_selling_price.png
│   ├── seller_type_distribution.png
│   ├── selling_price_boxplot.png
│   ├── selling_price_distribution.png
│   ├── transmission_distribution.png
│   ├── year_boxplot.png
│   └── year_vs_selling_price.png
│
├── best_car_price_model.pkl
├── DataScience-Task3-CarPricePrediction.ipynb
├── DataScience-Task3-CarPricePrediction.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

1. Clone the repository.

2. Install the required libraries.

```bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook or Python script.

---

## 📊 Project Outputs

The project generates:

- Distribution Plots
- Correlation Heatmap
- Scatter Plots
- Categorical Distribution Charts
- Box Plots
- Model Comparison Chart
- Actual vs Predicted Plot
- Trained Machine Learning Model

---

## 🚀 Future Improvements

- Hyperparameter Tuning
- Feature Selection
- Cross Validation
- Model Deployment using Flask or Streamlit
- Real-Time Car Price Prediction Web Application

---

## 👨‍💻 Author

**Shaikh Danish Shaikh Umar Farooq.**

**Oasis Infobyte Data Science Internship**