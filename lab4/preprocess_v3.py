# preprocess_v3.py
import pandas as pd

print("🔄 Создаём data_v3: One-Hot Encoding для Sex...")

# Загружаем версию 2
df = pd.read_csv('data_v2.csv')

# One-Hot Encoding для Sex
df = pd.get_dummies(df, columns=['Sex'], prefix='Sex', drop_first=False)

# Сбрасываем индекс
df = df.reset_index(drop=True)

# Сохраняем
df.to_csv('data_v3.csv', index=False)
print(f"✅ Готово: data_v3.csv ({len(df)} строк, колонки: {list(df.columns)})")