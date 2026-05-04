import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess():
    # Создаём необходимые папки
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    train = pd.read_csv("data/raw/train.csv")
    test = pd.read_csv("data/raw/test.csv")
    
    # Разделение на признаки и таргет
    X_train = train.drop("MedHouseVal", axis=1)
    y_train = train["MedHouseVal"]
    X_test = test.drop("MedHouseVal", axis=1)
    y_test = test["MedHouseVal"]
    
    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Сохранение обработанных данных
    np.save("data/processed/X_train.npy", X_train_scaled)
    np.save("data/processed/y_train.npy", y_train.values)
    np.save("data/processed/X_test.npy", X_test_scaled)
    np.save("data/processed/y_test.npy", y_test.values)
    
    # Сохранение скалера
    joblib.dump(scaler, "models/scaler.pkl")
    
    print("Данные предобработаны и сохранены")

if __name__ == "__main__":
    preprocess()