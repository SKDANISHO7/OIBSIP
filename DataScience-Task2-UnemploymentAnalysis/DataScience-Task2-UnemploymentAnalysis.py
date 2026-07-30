# 📊 Improved Unemployment Analysis in India
# Tech Stack: Python, pandas, matplotlib, seaborn

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Step 1: Load Dataset Safely
# -------------------------------
def load_dataset(filename="Unemployment.csv"):
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"❌ Dataset not found: {filename}\n"
            f"👉 Please place the CSV file in {os.getcwd()} or update the path."
        )
    df = pd.read_csv(filename)
    # Clean column names (strip spaces, shorten names)
    df.columns = df.columns.str.strip()
    df.rename(columns={
        "Estimated Unemployment Rate (%)": "UnemploymentRate",
        "Estimated Employed": "Employed",
        "Estimated Labour Participation Rate (%)": "LabourParticipationRate"
    }, inplace=True)
    return df

df = load_dataset()

print("✅ Dataset Loaded and Cleaned")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# -------------------------------
# Step 2: Data Cleaning
# -------------------------------
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)

print("Null values:\n", df.isnull().sum())

# -------------------------------
# Helper Function for Plot Styling
# -------------------------------
def plot_chart(figsize=(10,6)):
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

# -------------------------------
# Step 3: Region-wise Average Unemployment
# -------------------------------
region_avg = df.groupby('Region')['UnemploymentRate'].mean().sort_values(ascending=False)

plot_chart()
sns.barplot(x=region_avg.index, y=region_avg.values)
plt.xticks(rotation=90)
plt.title("Average Unemployment Rate by Region")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# -------------------------------
# Step 4: Month-wise Trends
# -------------------------------
df['Month'] = df['Date'].dt.month
month_avg = df.groupby('Month')['UnemploymentRate'].mean()

plot_chart(figsize=(8,5))
sns.lineplot(x=month_avg.index, y=month_avg.values, marker="o")
plt.title("Month-wise Average Unemployment Rate")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# -------------------------------
# Step 5: Time-Series for Major States
# -------------------------------
major_states = ['Maharashtra', 'Delhi', 'Tamil Nadu']
plot_chart(figsize=(12,6))

for state in major_states:
    state_data = df[df['Region'] == state]
    sns.lineplot(x=state_data['Date'], y=state_data['UnemploymentRate'], label=state)

plt.title("Unemployment Rate Over Time (Major States)")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# Step 6: Top 10 States
# -------------------------------
top10 = region_avg.head(10)

plot_chart()
sns.barplot(x=top10.index, y=top10.values)
plt.xticks(rotation=90)
plt.title("Top 10 States with Highest Average Unemployment")
plt.ylabel("Unemployment Rate (%)")
plt.tight_layout()
plt.show()

# -------------------------------
# Step 7: Heatmap (Correlation)
# -------------------------------
plot_chart(figsize=(8,6))
sns.heatmap(df[['UnemploymentRate','Employed','LabourParticipationRate']].corr(),
            annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# -------------------------------
# Step 8: Pre-COVID vs Post-COVID
# -------------------------------
pre_covid = df[df['Date'] < '2020-03-01']
post_covid = df[df['Date'] >= '2020-03-01']

print("📉 Pre-COVID Mean Unemployment:", round(pre_covid['UnemploymentRate'].mean(), 2))
print("📈 Post-COVID Mean Unemployment:", round(post_covid['UnemploymentRate'].mean(), 2))
