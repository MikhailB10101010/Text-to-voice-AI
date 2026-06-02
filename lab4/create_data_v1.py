# create_data_v1.py
import pandas as pd
import os

print("📥 Загружаем Titanic dataset...")

# Приоритет 1: Попробовать скачать с GitHub (стабильнее OpenML)
github_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

try:
    print("🔄 Пробуем скачать с GitHub...")
    df = pd.read_csv(github_url)
    print(f"✅ Загружено с GitHub: {len(df)} строк")
    
except Exception as e:
    print(f"⚠️ Не удалось скачать: {e}")
    
    # Приоритет 2: Проверить локальный файл
    if os.path.exists('titanic_manual.csv'):
        print("📁 Найден локальный файл titanic_manual.csv")
        df = pd.read_csv('titanic_manual.csv')
    else:
        # Приоритет 3: Минимальный fallback-датасет
        print("⚠️ Создаём минимальный датасет для продолжения работы с DVC")
        df = pd.DataFrame({
            'Pclass': [1,2,3,1,2,3,1,2],
            'Sex': ['male','female','male','female','male','female','male','female'],
            'Age': [22,38,26,35,28,40,18,50],
            'Survived': [0,1,0,1,0,0,1,0]
        })

# Приводим к нужному формату: оставляем только 4 колонки
required_cols = ['Pclass', 'Sex', 'Age', 'Survived']
df = df[required_cols].copy()

# Сохраняем
df.to_csv('data_v1.csv', index=False)
print(f"💾 Сохранено: data_v1.csv ({len(df)} строк)")
print(f"📊 Пример:\n{df.head(3)}")