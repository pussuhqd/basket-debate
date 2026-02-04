# Быстрая проверка
import pandas as pd

df = pd.read_csv('data/raw/russian_supermarket_prices.csv')
print("Колонки CSV:")
print(df.columns.tolist())
print("\nПример строки с яйцами:")
eggs = df[df['product_name'].str.contains('Яйца куриные', case=False, na=False)].iloc[0]
print(eggs)
