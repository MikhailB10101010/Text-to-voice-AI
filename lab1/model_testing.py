from pathlib import Path
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# Расположение
base_path = Path(__file__).parent
test_folder_name = 'test'
test_folder_path = base_path / test_folder_name
test_file = test_folder_path / 'test_preprocessed.csv'
model_folder_path = base_path / 'models'
model_name = 'model.pkl'
model_file = model_folder_path / model_name

# Загрузка тестовых данных и модели
if not test_file.exists():
    raise FileNotFoundError("Тестовые данные не найдены.")
if not model_file.exists():
    raise FileNotFoundError("Нет файла с моделью.")

test_df = pd.read_csv(test_file)
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Предсказание, точности
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Вывод результата в stdout (единственная строка вывода, как требуется)
print(f"Model test accuracy is: {accuracy:.3f}")
