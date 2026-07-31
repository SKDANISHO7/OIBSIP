# 🚗 Car Price Prediction

## 📌 Oasis Infobyte Data Science Internship - Task 3

## 📖 Project Overview

This project focuses on building a complete Machine Learning regression pipeline to predict the selling price of used cars based on various features such as car brand, age, mileage, fuel type, seller type, transmission, and ownership details.

The project includes complete data preprocessing, exploratory data analysis, feature engineering, model training, performance evaluation, and saving the best-performing regression model.

---

# 🎯 Objective

The objective of this project is to develop a reliable machine learning model that can accurately predict used car prices and identify the best regression algorithm based on evaluation metrics.

---

# 📂 Project Structure

```
DataScience-Task3-CarPricePrediction/
│
├── dataset/
│   └── car_data.csv
│
├── outputs/
│   ├── selling_price_distribution.png
│   ├── price_vs_fuel_type.png
│   ├── price_vs_year.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── actual_vs_predicted.png
│
├── DataScience-Task3-CarPricePrediction.ipynb
├── DataScience-Task3-CarPricePrediction.py
├── best_car_price_model.pkl
├── requirements.txt
└── README.md
```

---

# 📊 Dataset Information

The dataset contains information about used cars and their selling prices.

## Features Used

* Car_Name
* Year
* Present_Price
* Kms_Driven
* Fuel_Type
* Seller_Type
* Transmission
* Owner
* Selling_Price

---

# ⚙️ Feature Engineering

Additional features were created to improve model performance:

* **Brand Extraction**

  * Extracted car brand information from car names.

* **Car Age Calculation**

  * Calculated vehicle age using manufacturing year.

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Jupyter Notebook

---

# 🔍 Data Preprocessing

The following preprocessing steps were performed:

* Dataset loading and inspection
* Handling missing values
* Removing duplicate records
* Standardizing categorical values
* Feature extraction
* Encoding categorical variables using One-Hot Encoding
* Splitting dataset into training and testing sets

---

# 📈 Exploratory Data Analysis

The project includes:

* Selling price distribution analysis
* Price comparison based on fuel type
* Car price trend according to manufacturing year
* Correlation heatmap
* Feature importance visualization
* Actual vs predicted price comparison

All visualizations are automatically saved inside the `outputs/` folder.

---

# 🤖 Machine Learning Models

Multiple regression algorithms were implemented and compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

---

# 📏 Model Evaluation

Models were evaluated using:

* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**
* **R² Score**

The best-performing model was selected and saved as:

```
best_car_price_model.pkl
```

---

# 📁 Output Files Generated

The project generates the following output files:

```
outputs/
│
├── selling_price_distribution.png
├── price_vs_fuel_type.png
├── price_vs_year.png
├── correlation_heatmap.png
├── feature_importance.png
└── actual_vs_predicted.png
```

---

# ▶️ How to Run the Project

### 1. Clone Repository

```bash
git clone <repository-link>
```

### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 3. Run Python Script

```bash
python DataScience-Task3-CarPricePrediction.py
```

or open:

```
Car_Price_Prediction.ipynb
```

in Jupyter Notebook.

---

# ✅ Results

* Successfully developed a regression-based car price prediction system.
* Compared multiple machine learning models.
* Generated performance metrics and visual analysis.
* Saved the best trained model for future predictions.

---

# 👨‍💻 Author

**Shaikh Danish Shaikh Umar Farooq**

### Oasis Infobyte Data Science Internship

**Task 3 - Car Price Prediction**

---
