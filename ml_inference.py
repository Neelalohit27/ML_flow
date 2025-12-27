import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris

# Load Iris sample
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Load from MODEL REGISTRY
# Use version: models:/ModelName/Version
model_uri = "models:/Iris_RF_Model/1"     

loaded_model = mlflow.sklearn.load_model(model_uri)

# Make predictions
sample = X.iloc[:5]
pred = loaded_model.predict(sample)

print("Input Sample:")
print(sample)
print("\nPredictions:", pred)
