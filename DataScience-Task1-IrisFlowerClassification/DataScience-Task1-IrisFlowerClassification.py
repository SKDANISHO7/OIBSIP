# ==========================================================
# Iris Flower Classification
# Oasis Infobyte Internship - Task 1
# Author : Shaikh Danish
# ==========================================================

# ==========================================================
# Import Required Libraries
# ==========================================================

import os
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Ignore unnecessary warnings
warnings.filterwarnings("ignore")

# ==========================================================
# Project Configuration
# ==========================================================

OUTPUT_DIR = "outputs"
MODEL_FILE = "iris_best_model.pkl"

# Create Outputs Folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Start Timer
# ==========================================================

start_time = time.time()

# ==========================================================
# Visualization Settings
# ==========================================================

sns.set_theme(
    style="whitegrid",
    palette="Set2",
    context="notebook"
)

plt.rcParams.update({

    "figure.figsize": (8, 5),
    "figure.dpi": 120,
    "savefig.dpi": 300,

    "axes.titlesize": 15,
    "axes.titleweight": "bold",

    "axes.labelsize": 12,
    "axes.labelweight": "bold",

    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    "legend.fontsize": 10,

    "grid.alpha": 0.4
})

print("=" * 70)
print("IRIS FLOWER CLASSIFICATION PROJECT")
print("=" * 70)

print("Libraries Imported Successfully.")
print(f"Output Directory : {OUTPUT_DIR}")
print()

# ==========================================================
# Step 1 : Load Iris Dataset
# ==========================================================

print("=" * 70)
print("STEP 1 : LOADING DATASET")
print("=" * 70)

# Load Dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add Target Labels
df["species"] = [
    iris.target_names[target]
    for target in iris.target
]

# Store Feature Names
features = iris.feature_names

print("Dataset Loaded Successfully.")
print(f"Dataset Shape : {df.shape}")
print()

# ==========================================================
# Step 2 : Exploratory Data Analysis (EDA)
# ==========================================================

print("=" * 70)
print("STEP 2 : EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# -------------------------
# First Five Rows
# -------------------------

print("\nFirst Five Rows")
print("-" * 70)
print(df.head())

# -------------------------
# Dataset Information
# -------------------------

print("\nDataset Information")
print("-" * 70)
df.info()

# -------------------------
# Missing Values
# -------------------------

print("\nMissing Values")
print("-" * 70)
print(df.isnull().sum())

# -------------------------
# Statistical Summary
# -------------------------

print("\nStatistical Summary")
print("-" * 70)
print(df.describe().round(2))

# -------------------------
# Class Distribution
# -------------------------

print("\nSpecies Distribution")
print("-" * 70)
print(df["species"].value_counts())

# -------------------------
# Unique Species
# -------------------------

print("\nAvailable Classes")
print("-" * 70)

for index, species in enumerate(iris.target_names, start=1):
    print(f"{index}. {species}")

# ==========================================================
# Dataset Summary
# ==========================================================

print("\nDataset Summary")
print("-" * 70)

print(f"Total Samples        : {len(df)}")
print(f"Total Features       : {len(features)}")
print(f"Total Classes        : {df['species'].nunique()}")
print(f"Feature Names        : {', '.join(features)}")

print("\nEDA Completed Successfully.")
print()

# ==========================================================
# Step 3 : Data Visualization
# ==========================================================

print("=" * 70)
print("STEP 3 : DATA VISUALIZATION")
print("=" * 70)

# ==========================================================
# Histogram of Features
# ==========================================================

print("Generating Histograms...")

fig = df[features].hist(
    figsize=(12, 8),
    bins=15,
    edgecolor="black",
    color="skyblue"
)

plt.suptitle(
    "Histogram of Iris Features",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    os.path.join(OUTPUT_DIR, "histograms.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("✔ Histograms Saved")


# ==========================================================
# Pairplot
# ==========================================================

print("Generating Pairplot...")

pair = sns.pairplot(
    df,
    hue="species",
    palette="Set2",
    markers=["o", "s", "D"],
    diag_kind="hist",
    corner=False
)

pair.fig.suptitle(
    "Pairplot of Iris Dataset",
    fontsize=18,
    fontweight="bold",
    y=1.02
)

pair.savefig(
    os.path.join(OUTPUT_DIR, "pairplot.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("✔ Pairplot Saved")


# ==========================================================
# Correlation Heatmap
# ==========================================================

print("Generating Correlation Heatmap...")

plt.figure(figsize=(8, 6))

sns.heatmap(
    df[features].corr(),
    annot=True,
    cmap="coolwarm",
    square=True,
    linewidths=1,
    fmt=".2f",
    cbar=True
)

plt.title(
    "Correlation Heatmap",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "heatmap.png"),
    dpi=300,
    bbox_inches="tight"
)
# ==========================================================
# Step 4 : Boxplots of All Features
# ==========================================================

print("=" * 70)
print("STEP 4 : FEATURE DISTRIBUTION USING BOXPLOTS")
print("=" * 70)

print("Generating Boxplots...")

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8)
)

axes = axes.flatten()

for ax, feature in zip(axes, features):

    sns.boxplot(
        data=df,
        x="species",
        y=feature,
        hue="species",
        palette="Set2",
        legend=False,
        ax=ax
    )

    ax.set_title(
        feature.replace(" (cm)", "").title(),
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel("Species")
    ax.set_ylabel("Measurement (cm)")

fig.suptitle(
    "Distribution of Iris Features by Species",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    os.path.join(OUTPUT_DIR, "boxplots.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("✔ Boxplots Saved Successfully.")
print()


# ==========================================================
# Step 5 : Feature Selection Insights
# ==========================================================

print("=" * 70)
print("STEP 5 : FEATURE SELECTION INSIGHTS")
print("=" * 70)

feature_summary = pd.DataFrame({
    "Feature": features,
    "Correlation With Others": [
        df[feature].corr(df["petal length (cm)"])
        for feature in features
    ]
})

print(feature_summary.round(3))

print("\nImportant Observations")
print("-" * 70)

print("• Petal Length is the most discriminative feature.")
print("• Petal Width strongly separates flower species.")
print("• Sepal Length provides moderate classification ability.")
print("• Sepal Width contributes the least compared to other features.")

print("\nSelected Features for Model Training")
print("-" * 70)

for index, feature in enumerate(features, start=1):
    print(f"{index}. {feature}")

print("\nFeature Selection Completed Successfully.\n")

plt.show()
plt.close()

print("✔ Heatmap Saved")

print("\nData Visualization Completed Successfully.\n")

# ==========================================================
# Step 6 : Train-Test Split
# ==========================================================

print("=" * 70)
print("STEP 6 : TRAIN - TEST SPLIT")
print("=" * 70)

# Features and Target
X = df[features]
y = df["species"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")
print(f"Training Features: {X_train.shape[1]}")
print()

# ==========================================================
# Step 7 : Machine Learning Models
# ==========================================================

print("=" * 70)
print("STEP 7 : MODEL TRAINING")
print("=" * 70)

models = {

    "Logistic Regression":
    LogisticRegression(
        max_iter=300,
        random_state=42
    ),

    "K-Nearest Neighbors":
    KNeighborsClassifier(
        n_neighbors=5
    ),

    "Decision Tree":
    DecisionTreeClassifier(
        random_state=42
    )

}

results = {}

predictions = {}

trained_models = {}

# ==========================================================
# Train All Models
# ==========================================================

for name, model in models.items():

    print("=" * 70)
    print(f"Training : {name}")
    print("=" * 70)

    # Train Model
    model.fit(
        X_train,
        y_train
    )

    # Prediction
    y_pred = model.predict(
        X_test
    )

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # Store Results
    results[name] = accuracy
    predictions[name] = y_pred
    trained_models[name] = model

    print(f"Accuracy : {accuracy * 100:.2f}%\n")

    # Classification Report
    print("Classification Report")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

print("=" * 70)
print("ALL MODELS TRAINED SUCCESSFULLY")
print("=" * 70)
print()

# ==========================================================
# Step 8 : Confusion Matrix Visualization
# ==========================================================

print("=" * 70)
print("STEP 8 : CONFUSION MATRIX")
print("=" * 70)

for model_name, y_pred in predictions.items():

    print(f"Generating Confusion Matrix : {model_name}")

    # Confusion Matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=iris.target_names
    )

    # Accuracy
    accuracy = results[model_name]

    # Figure
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        linewidths=1,
        linecolor="white",
        square=True,
        cbar=True,
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )

    plt.title(
        f"{model_name}\nConfusion Matrix\nAccuracy : {accuracy*100:.2f}%",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel(
        "Predicted Class",
        fontsize=11,
        fontweight="bold"
    )

    plt.ylabel(
        "Actual Class",
        fontsize=11,
        fontweight="bold"
    )

    plt.tight_layout()

    # File Name
    file_name = (
        model_name.lower()
        .replace(" ", "_")
        .replace("-", "")
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"confusion_matrix_{file_name}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print(
        f"Saved : confusion_matrix_{file_name}.png"
    )

print()
print("=" * 70)
print("ALL CONFUSION MATRICES GENERATED SUCCESSFULLY")
print("=" * 70)
print()

# ==========================================================
# Step 9 : Model Comparison
# ==========================================================

print("=" * 70)
print("STEP 9 : MODEL COMPARISON")
print("=" * 70)

# Find Best Model
best_model_name = max(results, key=results.get)
best_accuracy = results[best_model_name]

# Create DataFrame
comparison_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": list(results.values())
})

# Sort by Accuracy
comparison_df = comparison_df.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)

# Plot
plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=comparison_df,
    x="Model",
    y="Accuracy",
    hue="Model",
    palette="Set2",
    legend=False
)

# Highlight Best Model
for bar, model in zip(ax.patches, comparison_df["Model"]):

    if model == best_model_name:
        bar.set_edgecolor("black")
        bar.set_linewidth(3)

# Accuracy Labels
for bar in ax.patches:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.003,
        f"{height*100:.2f}%",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

# Best Model Annotation
best_index = comparison_df[
    comparison_df["Model"] == best_model_name
].index[0]

plt.annotate(
    "Best Model",
    xy=(best_index, best_accuracy),
    xytext=(best_index, best_accuracy + 0.02),
    ha="center",
    fontsize=12,
    fontweight="bold",
    color="darkgreen",
    arrowprops=dict(
        arrowstyle="->",
        lw=2,
        color="green"
    )
)

plt.title(
    "Machine Learning Model Comparison",
    fontsize=17,
    fontweight="bold",
    pad=15
)

plt.xlabel(
    "Machine Learning Models",
    fontsize=12,
    fontweight="bold"
)

plt.ylabel(
    "Accuracy Score",
    fontsize=12,
    fontweight="bold"
)

plt.ylim(0.90, 1.05)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

# Save Figure
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print(f"Best Model : {best_model_name}")
print(f"Accuracy   : {best_accuracy*100:.2f}%")
print("Model Comparison Graph Saved Successfully.\n")

# ==========================================================
# Step 10 : Best Model & Feature Importance
# ==========================================================

print("=" * 70)
print("STEP 10 : BEST MODEL & FEATURE IMPORTANCE")
print("=" * 70)

# ----------------------------------------------------------
# Select Best Model
# ----------------------------------------------------------

best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]
best_accuracy = results[best_model_name]

print(f"Best Model      : {best_model_name}")
print(f"Accuracy Score  : {best_accuracy * 100:.2f}%")

# ----------------------------------------------------------
# Feature Importance (Decision Tree)
# ----------------------------------------------------------

print("\nGenerating Feature Importance Graph...")

tree_model = trained_models["Decision Tree"]

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": tree_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature",
    hue="Feature",
    palette="viridis",
    legend=False
)

# Add Importance Values
for index, value in enumerate(importance_df["Importance"]):

    plt.text(
        value + 0.005,
        index,
        f"{value:.3f}",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

plt.title(
    "Decision Tree Feature Importance",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel(
    "Importance Score",
    fontsize=12,
    fontweight="bold"
)

plt.ylabel(
    "Features",
    fontsize=12,
    fontweight="bold"
)

plt.grid(
    axis="x",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "feature_importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("Feature Importance Graph Saved Successfully.")

# ----------------------------------------------------------
# Save Best Model
# ----------------------------------------------------------

joblib.dump(
    best_model,
    MODEL_FILE
)

print(f"\nBest Model Saved Successfully : {MODEL_FILE}")
print()

# ==========================================================
# Step 11 : Sample Prediction
# ==========================================================

print("=" * 70)
print("STEP 11 : SAMPLE PREDICTION")
print("=" * 70)

# Select One Sample
sample = X_test.iloc[[0]]

# Actual Label
actual_label = y_test.iloc[0]

# Prediction
predicted_label = best_model.predict(sample)[0]

print("Sample Features")
print("-" * 70)
print(sample)

print("\nPrediction Result")
print("-" * 70)
print(f"Predicted Species : {predicted_label}")
print(f"Actual Species    : {actual_label}")

if predicted_label == actual_label:
    print("\nPrediction Status : Correct")
else:
    print("\nPrediction Status : Incorrect")


# ==========================================================
# Step 12 : Save Best Model
# ==========================================================

print("\n" + "=" * 70)
print("STEP 12 : SAVING BEST MODEL")
print("=" * 70)

MODEL_FILE = "iris_best_model.pkl"

joblib.dump(
    best_model,
    MODEL_FILE
)

print(f"Best Model Saved Successfully : {MODEL_FILE}")


# ==========================================================
# Step 13 : Execution Time & Project Summary
# ==========================================================

end_time = time.time()
execution_time = end_time - start_time

print("\n" + "=" * 70)
print("IRIS FLOWER CLASSIFICATION PROJECT SUMMARY")
print("=" * 70)

print("\nDataset Information")
print("-" * 70)
print(f"Dataset Name           : Iris Flower Dataset")
print(f"Total Samples          : {len(df)}")
print(f"Number of Features     : {len(features)}")
print(f"Number of Classes      : {df['species'].nunique()}")

print("\nMachine Learning Models Performance")
print("-" * 70)

for model_name, accuracy in results.items():
    print(f"{model_name:<25} : {accuracy*100:.2f}%")

print("\nBest Performing Model")
print("-" * 70)
print(f"Model Name             : {best_model_name}")
print(f"Accuracy Score         : {best_accuracy*100:.2f}%")

print("\nGenerated Output Files")
print("-" * 70)

generated_files = [
    "histograms.png",
    "pairplot.png",
    "heatmap.png",
    "boxplots.png",
    "confusion_matrix_logistic_regression.png",
    "confusion_matrix_knearest_neighbors.png",
    "confusion_matrix_decision_tree.png",
    "model_comparison.png",
    "feature_importance.png"
]

for i, file in enumerate(generated_files, start=1):
    print(f"{i}. {file}")

print("\nSaved Machine Learning Model")
print("-" * 70)
print(MODEL_FILE)

print("\nExecution Details")
print("-" * 70)
print(f"Total Execution Time   : {execution_time:.2f} seconds")

print("\nProject Status")
print("-" * 70)
print("Project Completed Successfully.")
print("Ready for GitHub Upload.")
print("Ready for Oasis Infobyte Submission.")

print("=" * 70)
print("THANK YOU")
print("=" * 70)