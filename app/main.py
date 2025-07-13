# app/main.py
'''importing fastapi for API utlization, 
pydantic for checking incoming request matches the schema defined,
joblib to load and save the trained model,
numpy for matrix operation as machine understand numerical data
'''
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Loading trained model
model = joblib.load("model/iris_model.pkl")
iris_classes = ['setosa', 'versicolor', 'virginica']

# Define FastAPI app
app = FastAPI(title="Iris Flower Prediction API")

# Define input schema so our input remains streamlined
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# root API , by default app response with this
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Prediction API"}

# predict API for predicting speices based on features
@app.post("/predict")
def predict_species(features: IrisFeatures):
    try:
        data = np.array([[features.sepal_length, features.sepal_width,
                          features.petal_length, features.petal_width]])
        pred = model.predict(data)[0]
        return {"prediction": iris_classes[pred]}
    except Exception as e:
        return {"error": str(e)}

