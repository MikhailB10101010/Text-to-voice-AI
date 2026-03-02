from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

#
base_path = Path(__file__).parent
train_folder_path = base_path / 'train'
test_folder_path = base_path / 'test'
train_file = train_folder_path / 'train_data.csv'
test_file = test_folder_path / 'test_data.csv'
train_preproc_file = train_folder_path / 'train_preprocessed.csv'
test_preproc_file = test_folder_path / 'test_preprocessed.csv'

# Проверяем наличие папок и файлов
if not (train_file.exists() and test_file.exists()):
    raise FileNotFoundError("Данные не найдены. Запустите data_creation.py")

# Загрузка данных
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

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
train_scaled_df.to_csv(train_preproc_file, index=False)
test_scaled_df.to_csv(test_preproc_file, index=False)
