import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_model():
    os.makedirs("models", exist_ok=True)
    
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    joblib.dump(model, "models/model.pkl")
    print(f"[OK] Model trained: coef={model.coef_[:3]}...")

if __name__ == "__main__":
    train_model()
