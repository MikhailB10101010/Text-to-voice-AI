import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import os

# Загрузка тестовых данных и модели
if not os.path.exists("test/test_preprocessed.csv"):
    raise FileNotFoundError("Тестовые данные не найдены.")

test_df = pd.read_csv("test/test_preprocessed.csv")
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

if not os.path.exists('model.pkl'):
    raise FileNotFoundError("Модель не найдена.")

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Предсказание
y_pred = model.predict(X_test)

# Оценка точности
accuracy = accuracy_score(y_test, y_pred)

# Вывод результата в stdout (единственная строка вывода, как требуется)
print(f"Model test accuracy is: {accuracy:.3f}")