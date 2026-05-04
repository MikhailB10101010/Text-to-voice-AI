from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
import os

app = FastAPI(title="ML Model Microservice Lab3")

# Загрузка модели и scaler
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

class PredictionInput(BaseModel):
    features: list  # Список признаков для предсказания (8 признаков California Housing)

@app.post("/predict")
def predict(input_data: PredictionInput):
    """Эндпоинт для получения предсказаний модели"""
    try:
        features = np.array(input_data.features).reshape(1, -1)
        # Масштабирование данных перед предсказанием
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return {
            "prediction": float(prediction[0]),
            "model_type": type(model).__name__
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    """Проверка работоспособности сервиса"""
    return {"status": "healthy", "model_loaded": model is not None, "scaler_loaded": scaler is not None}
