"""
Добавляет mock свежие продукты в БД с embeddings.
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


DB_PATH = Path('data/processed/products.db')

# Mock свежие продукты
MOCK_PRODUCTS = [
    {
        "id": 900001,
        "product_name": "Яйца куриные С1",
        "product_category": "Яйца",
        "brand": "Окское",
        "price_per_unit": 89.90,
        "unit": "уп",
        "package_size": 10,
        "tags": "eggs,protein",
        "meal_components": ["main_course", "breakfast"]
    },
    {
        "id": 900002,
        "product_name": "Помидоры свежие",
        "product_category": "Овощи и фрукты",
        "brand": "Местные",
        "price_per_unit": 149.90,
        "unit": "кг",
        "package_size": 1.0,
        "tags": "vegetables,fresh",
        "meal_components": ["salad", "side_dish"]
    },
    {
        "id": 900003,
        "product_name": "Огурцы свежие",
        "product_category": "Овощи и фрукты",
        "brand": "Местные",
        "price_per_unit": 129.90,
        "unit": "кг",
        "package_size": 1.0,
        "tags": "vegetables,fresh",
        "meal_components": ["salad", "side_dish"]
    },
    {
        "id": 900004,
        "product_name": "Картофель белый",
        "product_category": "Овощи и фрукты",
        "brand": "Местные",
        "price_per_unit": 49.90,
        "unit": "кг",
        "package_size": 2.5,
        "tags": "vegetables,fresh",
        "meal_components": ["side_dish"]
    },
    {
        "id": 900005,
        "product_name": "Морковь мытая",
        "product_category": "Овощи и фрукты",
        "brand": "Местные",
        "price_per_unit": 39.90,
        "unit": "кг",
        "package_size": 1.0,
        "tags": "vegetables,fresh",
        "meal_components": ["side_dish", "salad"]
    },
    {
        "id": 900006,
        "product_name": "Лук репчатый",
        "product_category": "Овощи и фрукты",
        "brand": "Местные",
        "price_per_unit": 29.90,
        "unit": "кг",
        "package_size": 1.0,
        "tags": "vegetables,fresh",
        "meal_components": ["side_dish"]
    },
    {
        "id": 900007,
        "product_name": "Перец болгарский красный",
        "product_category": "Овощи и фрукты",
        "brand": "Местные",
        "price_per_unit": 199.90,
        "unit": "кг",
        "package_size": 1.0,
        "tags": "vegetables,fresh",
        "meal_components": ["salad", "side_dish"]
    },
    {
        "id": 900008,
        "product_name": "Капуста белокочанная",
        "product_category": "Овощи и фрукты",
        "brand": "Местные",
        "price_per_unit": 35.90,
        "unit": "кг",
        "package_size": 1.5,
        "tags": "vegetables,fresh",
        "meal_components": ["salad", "side_dish"]
    },
    {
        "id": 900009,
        "product_name": "Кабачок",
        "product_category": "Овощи и фрукты",
        "brand": "Местные",
        "price_per_unit": 89.90,
        "unit": "кг",
        "package_size": 1.0,
        "tags": "vegetables,fresh",
        "meal_components": ["side_dish"]
    },
    {
        "id": 900010,
        "product_name": "Баклажан",
        "product_category": "Овощи и фрукты",
        "brand": "Местные",
        "price_per_unit": 129.90,
        "unit": "кг",
        "package_size": 1.0,
        "tags": "vegetables,fresh",
        "meal_components": ["side_dish"]
    },
    {
        "id": 900011,
        "product_name": "Молоко пастеризованное 3.2%",
        "product_category": "Молочные продукты",
        "brand": "Простоквашино",
        "price_per_unit": 89.90,
        "unit": "л",
        "package_size": 1.0,
        "tags": "dairy,fresh",
        "meal_components": ["beverage"]
    },
    {
        "id": 900012,
        "product_name": "Сметана 15%",
        "product_category": "Молочные продукты",
        "brand": "Простоквашино",
        "price_per_unit": 79.90,
        "unit": "г",
        "package_size": 300,
        "tags": "dairy",
        "meal_components": ["sauce"]
    },
    {
        "id": 900013,
        "product_name": "Творог 5%",
        "product_category": "Молочные продукты",
        "brand": "Простоквашино",
        "price_per_unit": 119.90,
        "unit": "г",
        "package_size": 300,
        "tags": "dairy,protein",
        "meal_components": ["breakfast"]
    },
    {
        "id": 900014,
        "product_name": "Масло подсолнечное рафинированное",
        "product_category": "Масло",
        "brand": "Слобода",
        "price_per_unit": 149.90,
        "unit": "л",
        "package_size": 1.0,
        "tags": "oil",
        "meal_components": ["sauce"]
    },
    {
        "id": 900015,
        "product_name": "Масло сливочное 82.5%",
        "product_category": "Масло",
        "brand": "Простоквашино",
        "price_per_unit": 189.90,
        "unit": "г",
        "package_size": 180,
        "tags": "dairy,oil",
        "meal_components": ["sauce"]
    }
]


def add_products_to_db():
    """Добавляет свежие продукты в БД."""
    
    print("=" * 70)
    print("🥗 ДОБАВЛЕНИЕ СВЕЖИХ ПРОДУКТОВ В БД")
    print("=" * 70)
    
    # 1. Загружаем модель для embeddings
    print("\n🔄 Загрузка модели...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    print("   ✅ Модель загружена")
    
    # 2. Подключаемся к БД
    print(f"\n📂 Подключение к {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 3. Добавляем mock продукты
    print(f"\n🥦 Добавление {len(MOCK_PRODUCTS)} продуктов...")
    
    added = 0
    updated = 0
    
    for product in MOCK_PRODUCTS:
        # Генерируем embedding
        text = f"{product['product_name']} {product['product_category']} {product.get('brand', '')}"
        embedding = model.encode(text, convert_to_numpy=True)
        embedding_blob = embedding.astype(np.float32).tobytes()
        
        # Проверяем существует ли
        cursor.execute("SELECT id FROM products WHERE id = ?", (product['id'],))
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute("""
                INSERT INTO products 
                (id, product_name, product_category, brand, price_per_unit, unit, 
                 package_size, tags, meal_components, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                product['id'],
                product['product_name'],
                product['product_category'],
                product['brand'],
                product['price_per_unit'],
                product['unit'],
                product['package_size'],
                product['tags'],
                json.dumps(product['meal_components']),
                embedding_blob
            ))
            print(f"   ✅ Добавлено: {product['product_name']}")
            added += 1
        else:
            # Обновляем embedding
            cursor.execute("""
                UPDATE products 
                SET embedding = ?
                WHERE id = ?
            """, (embedding_blob, product['id']))
            print(f"   🔄 Обновлено: {product['product_name']}")
            updated += 1
    
    # 4. Сохраняем
    conn.commit()
    conn.close()
    
    print("\n" + "=" * 70)
    print(f"✅ Добавлено: {added} | Обновлено: {updated}")
    print("=" * 70)
    
    # 5. Проверка
    print("\n🔍 Проверка добавленных продуктов...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    test_queries = ['помидор', 'огурец', 'яйц', 'картофель', 'морковь', 'масло']
    for query in test_queries:
        cursor.execute(f"""
            SELECT product_name, product_category, price_per_unit 
            FROM products 
            WHERE (product_name LIKE '%{query}%' OR product_category LIKE '%{query}%')
            AND id >= 900000
            LIMIT 3
        """)
        results = cursor.fetchall()
        if results:
            print(f"\n   '{query}': {len(results)} товаров")
            for name, cat, price in results:
                print(f"      - {name} ({cat}) - {price}₽")
    
    conn.close()


if __name__ == '__main__':
    add_products_to_db()
