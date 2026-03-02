import sys
from pathlib import Path
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

# Взятие рандома из .sh
if len(sys.argv) > 1:
    random_id = int(sys.argv[1])
else:
    random_id = 123

# Расположение
base_path = Path(__file__).parent
train_folder_name = 'train'
train_folder_path = base_path / train_folder_name
train_file = train_folder_path / 'train_preprocessed.csv'
model_folder_path = base_path / 'models'
model_file_name = 'model.pkl'

# Загрузка предобработанных данных
if not train_file.exists():
    raise FileNotFoundError("Предобработанные данные не найдены.")

train_df = pd.read_csv(train_file)

X_train = train_df.drop('target', axis=1)
y_train = train_df['target']

# Создание и обучение модели
model = LogisticRegression(random_state=random_id)
model.fit(X_train, y_train)

# Сохранение модели в файл
model_folder_path.mkdir(exist_ok=True)
with open(model_folder_path / model_file_name, 'wb') as f:
    pickle.dump(model, f)
