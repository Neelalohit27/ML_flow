# MLflow Experiment Tracking with Google Colab

This project demonstrates End-to-End Machine Learning Experiment Tracking using MLflow. It focuses on training multiple machine learning models, logging metrics, parameters, artifacts, and comparing model performance visually through the MLflow UI.
---

## 🚀 Project Objectives

1.Perform data preprocessing

2.Train multiple ML models

3.Log:

    -Accuracy

    -Confusion Matrix

4.Track experiments in **MLflow UI**

5.Compare and analyze performance between models
---
## 📌 Tech Stack

Python

Pandas / NumPy

Scikit-Learn

Matplotlib / Seaborn

MLflow

Google Colab

ngrok (for MLflow UI tunnel)
---
## 📂 Dataset

The dataset contains features and target used for classification.
Data preprocessing includes:
    Handling null values
    Label Encoding
    Train-Test split.
---
## 🛠️ Steps Performed in Code
### ✅ 1. Install Dependencies
```bash
pip install mlflow scikit-learn pandas numpy matplotlib seaborn pyngrok
```
### ✅ 2. Start MLflow Tracking Server

MLflow UI is hosted locally in Colab and accessed via ngrok public URL.
### ✅ 3. Train and Log Multiple Models

#### Model Trained:
Random Forest

#### MLflow logs:
✔ Parameters
✔ Metrics
✔ Confusion Matrix Plot
---
## 📎 How to Run

1️⃣ Create the directory structure 
2️⃣ Copy all the files into their respective named files
3️⃣ Open MLflow UI using generated public URL
4️⃣ View experiments and screenshots