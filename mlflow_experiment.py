
# MLflow Experiment Tracking Script (Windows-Ready)

import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.datasets import load_iris

warnings.filterwarnings("ignore")

# Set MLflow tracking URI & experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")   
mlflow.set_experiment("MLflow_Iris_Experiment")

# Create artifacts folder
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Load Dataset (Iris)
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to run experiment
def run_experiment(model, model_name, params=None):
    with mlflow.start_run(run_name=model_name) as run:
        
        # Log parameters
        if params:
            mlflow.log_params(params)

        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("model_type", "RandomForestClassifier")

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        preds = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)

        # Confusion Matrix
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title(f"{model_name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        cm_path = os.path.join(ARTIFACT_DIR, f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)

        # Log Model + Register Model
        mlflow.sklearn.log_model(model, model_name)

        model_uri = f"runs:/{run.info.run_id}/{model_name}"

        try:
            mlflow.register_model(
                model_uri=model_uri,
                name="Iris_RF_Model"
            )
            print("Model registered successfully.")
        except Exception as e:
            print("Model already registered or registry disabled:", e)

        print(f"Completed: {model_name} | Accuracy: {acc:.4f}")
        print(f"Artifact saved at: {cm_path}")


# Run Multiple Experiments
hyperparams_list = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 150, "max_depth": 7},
]

for params in hyperparams_list:
    model_name = f"RF_{params['n_estimators']}_{params['max_depth']}"
    rf = RandomForestClassifier(**params, random_state=42)
    run_experiment(rf, model_name, params)
