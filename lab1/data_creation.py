#
import sys
from pathlib import Path
#
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Взятие рандома из .sh
if len(sys.argv) > 1:
    random_id = int(sys.argv[1])
else:
    random_id = 123

#
base_path = Path(__file__).parent
train_folder_path = base_path / 'train'
test_folder_path = base_path / 'test'

# Создаем папки для данных
train_folder_path.mkdir(exist_ok=True)
test_folder_path.mkdir(exist_ok=True)

# Генерируем синтетический набор данных (3000 записей)
# Эмуляция процесса: например, показания датчиков и статус системы (0 - норма, 1 - сбой)
X, y = make_classification(
    n_samples=3000,
    n_features=5,
    n_informative=3,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=random_id
)

# Конвертируем в DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
df['target'] = y

# Добавляем "шум" и аномалии в случайные места
# Например, выбираем 50 случайных строк и заменяем значения признаков на случайные большие числа
n_anomalies = 50
anomaly_indices = np.random.choice(df.index, n_anomalies, replace=False)
for idx in anomaly_indices:
    df.loc[idx, 'feature_0'] = np.random.uniform(10, 20)  # Аномалия в 0-м признаке

# Разделение на train (обучающая) и test (тестовая)
# 80% на обучение, 20% на тест
# train_size = int(0.8 * len(df))
# train_df = df.iloc[:train_size]
# test_df = df.iloc[train_size:]
train_df = df.sample(frac=0.7, random_state=random_id)
test_df = df.drop(train_df.index)

# Сохранение файлов
train_df.to_csv(train_folder_path / "train_data.csv", index=False)
test_df.to_csv(test_folder_path / "test_data.csv", index=False)

# Можно создать несколько наборов, как в задании (для примера создадим копии с другим именем)
# train_df.sample(frac=0.5).to_csv("train/train_data_subset.csv", index=False)
