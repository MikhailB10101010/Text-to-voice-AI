import sys
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import os

# Взятие рандома из .sh
if len(sys.argv) > 1:
    random_id = int(sys.argv[1])
else:
    random_id = 123

# Загрузка предобработанных данных
if not os.path.exists("train/train_preprocessed.csv"):
    raise FileNotFoundError("Предобработанные данные не найдены.")

train_df = pd.read_csv("train/train_preprocessed.csv")

X_train = train_df.drop('target', axis=1)
y_train = train_df['target']

# Создание и обучение модели
model = LogisticRegression(random_state=random_id)
model.fit(X_train, y_train)

# Сохранение модели в файл
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
