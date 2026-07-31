# ==============================================================================
# Car Price Prediction
# Oasis Infobyte Data Science Internship - Task 3
# ==============================================================================

# Step 1 : Import Required Libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Step 2 : Create Outputs Folder
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# Step 3 : Load Dataset
df = pd.read_excel("Car details v3.xlsx")   # replace with your chosen file

# Step 4 : Dataset Overview
print(df.head())
print(df.info())
print(df.describe())

# Step 5 : Data Preprocessing
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Normalize categorical values
df['fuel'] = df['fuel'].str.lower().str.strip()
df['transmission'] = df['transmission'].str.lower().str.strip()

# Step 6 : Exploratory Data Analysis (EDA)
plt.figure(figsize=(8,5))
sns.histplot(df['selling_price'], bins=30, kde=True)
plt.title("Distribution of Selling Prices")
plt.savefig("outputs/price_distribution.png")
plt.show()

sns.boxplot(x='fuel', y='selling_price', data=df)
plt.title("Price vs Fuel Type")
plt.savefig("outputs/price_vs_fuel.png")
plt.show()

sns.scatterplot(x='year', y='selling_price', data=df)
plt.title("Price vs Year")
plt.savefig("outputs/price_vs_year.png")
plt.show()

# Step 7 : Feature Engineering
from datetime import datetime
df['car_age'] = datetime.now().year - df['year']
df['brand'] = df['car_name'].str.split().str[0]

# Encode categorical variables
df = pd.get_dummies(df, columns=['fuel','transmission','brand'], drop_first=True)

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

# Step 8 : Train-Test Split
X = df.drop(['selling_price','car_name'], axis=1)
y = df['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9 : Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Step 10 : Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Step 11 : Random Forest Regressor
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Step 12 : Model Comparison
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
    return mae, rmse, r2

results = {}
results['Linear Regression'] = evaluate_model("Linear Regression", y_test, y_pred_lr)
results['Decision Tree'] = evaluate_model("Decision Tree", y_test, y_pred_dt)
results['Random Forest'] = evaluate_model("Random Forest", y_test, y_pred_rf)

# Step 13 : Actual vs Predicted Plot (Best Model)
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_rf, alpha=0.7)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted (Random Forest)")
plt.savefig("outputs/actual_vs_predicted.png")
plt.show()

# Step 14 : Save Best Model
joblib.dump(rf, "outputs/best_model.pkl")
print("Best model saved as outputs/best_model.pkl")
