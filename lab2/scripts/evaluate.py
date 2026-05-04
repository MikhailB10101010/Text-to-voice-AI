import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def evaluate():
    model = joblib.load("models/model.pkl")
    
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model test MSE: {mse:.4f}")
    print(f"Model test R2: {r2:.4f}")
    
    # Для Jenkins: вывод одной строки с метрикой
    print(f"METRIC_R2={r2:.4f}")
    
    return r2

if __name__ == "__main__":
    evaluate()
