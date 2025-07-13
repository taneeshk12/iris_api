# Iris Flower Classifier API with FastAPI & Docker

This project demonstrates an end-to-end machine learning model deployment pipeline using **FastAPI**, **Logistic Regression**, and **Docker**. The API predicts the species of Iris flowers based on sepal and petal dimensions.

---

## 🚀 Features

* Trained on the **Iris dataset** using **Logistic Regression**
* Deployed as a REST API using **FastAPI**
* Input validation using **Pydantic**
* **Dockerized** for consistent deployment
* Compatible with free deployment platforms like **Render**, **Railway**, and **Fly.io**

---

## 📁 Project Structure

```
iris_api/
├── model/
│   └── iris_model.pkl         # Trained ML model
├── app/
│   ├── main.py                # FastAPI application
│   └── schemas.py (optional) # Pydantic model
├── train_model.py            # Script to train and save the model
├── requirements.txt          # Project dependencies
└── Dockerfile                # Docker setup for deployment
```

---

## 🔍 Tech Stack

* Python 3.10
* scikit-learn
* FastAPI
* Pydantic
* Docker
* Uvicorn

---

## 📦 Installation & Setup

```bash
# Clone the repository
https://github.com/your-username/iris-fastapi-api
cd iris-fastapi-api

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Train and save the model
python train_model.py

# Run the API
uvicorn app.main:app --reload
```

---

##  API Endpoint

### POST `/predict`

**Request Body:**

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Response:**

```json
{
  "prediction": "setosa"
}
```

---

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t iris-api .

# Run the container
docker run -d -p 8000:8000 iris-api

# Visit the API docs
http://localhost:8000/docs
```

---
