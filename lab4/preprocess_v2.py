# preprocess_v2.py
import pandas as pd

print("🔄 Создаём data_v2: заполняем пропуски в Age медианой по Pclass...")

# Загружаем исходную версию
df = pd.read_csv('data_v1.csv')

# Заполняем Age медианой внутри каждого Pclass
df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))

# Сохраняем новую версию
df.to_csv('data_v2.csv', index=False)
print(f"✅ Готово: data_v2.csv ({len(df)} строк, пропусков в Age: {df['Age'].isna().sum()})")