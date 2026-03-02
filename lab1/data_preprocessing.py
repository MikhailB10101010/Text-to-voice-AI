import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Проверяем наличие папок и файлов
if not os.path.exists("train/train_data.csv") or not os.path.exists("test/test_data.csv"):
    raise FileNotFoundError("Данные не найдены. Запустите data_creation.py")

# Загрузка данных
train_df = pd.read_csv("train/train_data.csv")
test_df = pd.read_csv("test/test_data.csv")

# Отделяем признаки от целевой переменной
X_train = train_df.drop('target', axis=1)
y_train = train_df['target']

X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

# Инициализация и обучение скалера ТОЛЬКО на обучающей выборке
scaler = StandardScaler()
scaler.fit(X_train)

# Трансформация обеих выборок
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Создание DataFrame обратно для удобства сохранения
train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
train_scaled_df['target'] = y_train.values

test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
test_scaled_df['target'] = y_test.values

# Сохранение предобработанных данных
train_scaled_df.to_csv("train/train_preprocessed.csv", index=False)
test_scaled_df.to_csv("test/test_preprocessed.csv", index=False)