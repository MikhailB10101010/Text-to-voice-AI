import pandas as pd
import os
from sklearn.datasets import fetch_california_housing

def fetch_and_save():
    os.makedirs("data/raw", exist_ok=True)
    
    # встроенный датасет
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    # Разделение на train/test
    train = df.sample(frac=0.8, random_state=42)
    test = df.drop(train.index)
    
    train.to_csv("data/raw/train.csv", index=False)
    test.to_csv("data/raw/test.csv", index=False)
    print(f"Данные сохранены: train={len(train)}, test={len(test)}")

if __name__ == "__main__":
    fetch_and_save()