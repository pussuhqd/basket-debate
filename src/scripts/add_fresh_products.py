"""
Создаёт полный набор базовых продуктов для демо MVP.
~80 товаров, покрывающих основные сценарии.
"""

import sqlite3
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


DB_PATH = Path('data/processed/products.db')

# Полный набор mock продуктов
MOCK_PRODUCTS = [
    # ========== КРУПЫ И МАКАРОНЫ ==========
    {"id": 900101, "name": "Овсянка Геркулес", "category": "Крупы", "brand": "Myllyn Paras", "price": 89.90, "unit": "кг", "size": 0.5, "tags": "cereals", "components": ["breakfast"]},
    {"id": 900102, "name": "Рис круглозерный", "category": "Крупы", "brand": "Мистраль", "price": 79.90, "unit": "кг", "size": 1.0, "tags": "cereals", "components": ["side_dish"]},
    {"id": 900103, "name": "Гречка ядрица", "category": "Крупы", "brand": "Makfa", "price": 99.90, "unit": "кг", "size": 0.8, "tags": "cereals", "components": ["side_dish"]},
    {"id": 900104, "name": "Макароны спагетти", "category": "Макароны", "brand": "Barilla", "price": 119.90, "unit": "г", "size": 500, "tags": "pasta", "components": ["main_course"]},
    {"id": 900105, "name": "Макароны пенне", "category": "Макароны", "brand": "Makfa", "price": 59.90, "unit": "г", "size": 400, "tags": "pasta", "components": ["main_course"]},
    
    # ========== ОВОЩИ ==========
    {"id": 900201, "name": "Помидоры свежие", "category": "Овощи", "brand": "Местные", "price": 149.90, "unit": "кг", "size": 1.0, "tags": "vegetables", "components": ["salad", "side_dish"]},
    {"id": 900202, "name": "Огурцы свежие", "category": "Овощи", "brand": "Местные", "price": 129.90, "unit": "кг", "size": 1.0, "tags": "vegetables", "components": ["salad"]},
    {"id": 900203, "name": "Картофель", "category": "Овощи", "brand": "Местные", "price": 49.90, "unit": "кг", "size": 2.5, "tags": "vegetables", "components": ["side_dish"]},
    {"id": 900204, "name": "Морковь", "category": "Овощи", "brand": "Местные", "price": 39.90, "unit": "кг", "size": 1.0, "tags": "vegetables", "components": ["side_dish", "salad"]},
    {"id": 900205, "name": "Лук репчатый", "category": "Овощи", "brand": "Местные", "price": 29.90, "unit": "кг", "size": 1.0, "tags": "vegetables", "components": ["side_dish"]},
    {"id": 900206, "name": "Перец болгарский", "category": "Овощи", "brand": "Местные", "price": 199.90, "unit": "кг", "size": 0.5, "tags": "vegetables", "components": ["salad"]},
    {"id": 900207, "name": "Капуста белокочанная", "category": "Овощи", "brand": "Местные", "price": 35.90, "unit": "кг", "size": 1.5, "tags": "vegetables", "components": ["salad"]},
    {"id": 900208, "name": "Кабачок", "category": "Овощи", "brand": "Местные", "price": 89.90, "unit": "кг", "size": 1.0, "tags": "vegetables", "components": ["side_dish"]},
    {"id": 900209, "name": "Баклажан", "category": "Овощи", "brand": "Местные", "price": 129.90, "unit": "кг", "size": 1.0, "tags": "vegetables", "components": ["side_dish"]},
    {"id": 900210, "name": "Свекла", "category": "Овощи", "brand": "Местные", "price": 45.90, "unit": "кг", "size": 1.0, "tags": "vegetables", "components": ["salad", "side_dish"]},
    
    # ========== ФРУКТЫ ==========
    {"id": 900301, "name": "Бананы", "category": "Фрукты", "brand": "Эквадор", "price": 89.90, "unit": "кг", "size": 1.0, "tags": "fruits", "components": ["snack"]},
    {"id": 900302, "name": "Яблоки Голден", "category": "Фрукты", "brand": "Россия", "price": 119.90, "unit": "кг", "size": 1.0, "tags": "fruits", "components": ["snack"]},
    {"id": 900303, "name": "Апельсины", "category": "Фрукты", "brand": "Марокко", "price": 139.90, "unit": "кг", "size": 1.0, "tags": "fruits", "components": ["snack"]},
    {"id": 900304, "name": "Мандарины", "category": "Фрукты", "brand": "Турция", "price": 149.90, "unit": "кг", "size": 1.0, "tags": "fruits", "components": ["snack"]},
    
    # ========== МЯСО И ПТИЦА ==========
    {"id": 900401, "name": "Куриное филе", "category": "Мясо", "brand": "Петелинка", "price": 389.90, "unit": "кг", "size": 1.0, "tags": "meat,protein", "components": ["main_course"]},
    {"id": 900402, "name": "Куриные бедра", "category": "Мясо", "brand": "Петелинка", "price": 249.90, "unit": "кг", "size": 1.0, "tags": "meat,protein", "components": ["main_course"]},
    {"id": 900403, "name": "Говядина вырезка", "category": "Мясо", "brand": "Мираторг", "price": 699.90, "unit": "кг", "size": 0.5, "tags": "meat,protein", "components": ["main_course"]},
    {"id": 900404, "name": "Свинина вырезка", "category": "Мясо", "brand": "Мираторг", "price": 449.90, "unit": "кг", "size": 0.6, "tags": "meat,protein", "components": ["main_course"]},
    {"id": 900405, "name": "Фарш говяжий", "category": "Мясо", "brand": "Мираторг", "price": 389.90, "unit": "кг", "size": 0.5, "tags": "meat,protein", "components": ["main_course"]},
    
    # ========== РЫБА ==========
    {"id": 900501, "name": "Филе семги", "category": "Рыба", "brand": "Норвегия", "price": 899.90, "unit": "кг", "size": 0.3, "tags": "fish,protein", "components": ["main_course"]},
    {"id": 900502, "name": "Минтай филе", "category": "Рыба", "brand": "Русское море", "price": 299.90, "unit": "кг", "size": 0.5, "tags": "fish,protein", "components": ["main_course"]},
    {"id": 900503, "name": "Тунец консервированный", "category": "Рыба", "brand": "Fortuna", "price": 189.90, "unit": "г", "size": 185, "tags": "fish", "components": ["main_course"]},
    
    # ========== МОЛОЧКА ==========
    {"id": 900601, "name": "Молоко 3.2%", "category": "Молочные продукты", "brand": "Простоквашино", "price": 89.90, "unit": "л", "size": 1.0, "tags": "dairy", "components": ["beverage"]},
    {"id": 900602, "name": "Кефир 2.5%", "category": "Молочные продукты", "brand": "Простоквашино", "price": 79.90, "unit": "л", "size": 1.0, "tags": "dairy", "components": ["beverage"]},
    {"id": 900603, "name": "Творог 5%", "category": "Молочные продукты", "brand": "Простоквашино", "price": 119.90, "unit": "г", "size": 300, "tags": "dairy,protein", "components": ["breakfast"]},
    {"id": 900604, "name": "Сметана 15%", "category": "Молочные продукты", "brand": "Простоквашино", "price": 79.90, "unit": "г", "size": 300, "tags": "dairy", "components": ["sauce"]},
    {"id": 900605, "name": "Йогурт натуральный", "category": "Молочные продукты", "brand": "Активиа", "price": 69.90, "unit": "г", "size": 350, "tags": "dairy", "components": ["breakfast"]},
    {"id": 900606, "name": "Сыр Российский", "category": "Сыр", "brand": "Киприно", "price": 499.90, "unit": "кг", "size": 1.0, "tags": "dairy", "components": ["snack"]},
    {"id": 900607, "name": "Сыр Пармезан", "category": "Сыр", "brand": "Grana Padano", "price": 899.90, "unit": "г", "size": 200, "tags": "dairy", "components": ["snack"]},
    
    # ========== ЯЙЦА ==========
    {"id": 900701, "name": "Яйца куриные С1", "category": "Яйца", "brand": "Окское", "price": 89.90, "unit": "уп", "size": 10, "tags": "eggs,protein", "components": ["breakfast", "main_course"]},
    
    # ========== ХЛЕБ И ВЫПЕЧКА ==========
    {"id": 900801, "name": "Хлеб белый нарезной", "category": "Хлеб", "brand": "Коломенское", "price": 49.90, "unit": "г", "size": 400, "tags": "bakery", "components": ["bakery"]},
    {"id": 900802, "name": "Хлеб черный", "category": "Хлеб", "brand": "Бородинский", "price": 59.90, "unit": "г", "size": 400, "tags": "bakery", "components": ["bakery"]},
    {"id": 900803, "name": "Батон нарезной", "category": "Хлеб", "brand": "Коломенское", "price": 45.90, "unit": "г", "size": 350, "tags": "bakery", "components": ["bakery"]},
    
    # ========== МАСЛО И СОУСЫ ==========
    {"id": 900901, "name": "Масло подсолнечное", "category": "Масло", "brand": "Слобода", "price": 149.90, "unit": "л", "size": 1.0, "tags": "oil", "components": ["sauce"]},
    {"id": 900902, "name": "Масло оливковое", "category": "Масло", "brand": "Borges", "price": 449.90, "unit": "мл", "size": 500, "tags": "oil", "components": ["sauce"]},
    {"id": 900903, "name": "Масло сливочное 82.5%", "category": "Масло", "brand": "Простоквашино", "price": 189.90, "unit": "г", "size": 180, "tags": "dairy,oil", "components": ["sauce"]},
    {"id": 900904, "name": "Майонез Провансаль", "category": "Соусы", "brand": "Слобода", "price": 119.90, "unit": "г", "size": 400, "tags": "sauce", "components": ["sauce"]},
    {"id": 900905, "name": "Кетчуп томатный", "category": "Соусы", "brand": "Heinz", "price": 139.90, "unit": "г", "size": 450, "tags": "sauce", "components": ["sauce"]},
    
    # ========== СПЕЦИИ И БАЗОВЫЕ ПРОДУКТЫ ==========
    {"id": 901001, "name": "Соль поваренная", "category": "Специи", "brand": "Экстра", "price": 19.90, "unit": "кг", "size": 1.0, "tags": "spices", "components": ["sauce"]},
    {"id": 901002, "name": "Сахар-песок", "category": "Сахар", "brand": "Русский", "price": 69.90, "unit": "кг", "size": 1.0, "tags": "sugar", "components": ["breakfast"]},
    {"id": 901003, "name": "Мука пшеничная высший сорт", "category": "Мука", "brand": "Makfa", "price": 59.90, "unit": "кг", "size": 2.0, "tags": "flour", "components": ["bakery"]},
    {"id": 901004, "name": "Перец черный молотый", "category": "Специи", "brand": "Kotanyi", "price": 89.90, "unit": "г", "size": 50, "tags": "spices", "components": ["sauce"]},
    
    # ========== НАПИТКИ ==========
    {"id": 901101, "name": "Сок апельсиновый", "category": "Напитки", "brand": "Добрый", "price": 99.90, "unit": "л", "size": 1.0, "tags": "beverages", "components": ["beverage"]},
    {"id": 901102, "name": "Чай черный", "category": "Напитки", "brand": "Lipton", "price": 199.90, "unit": "г", "size": 100, "tags": "beverages", "components": ["beverage"]},
    {"id": 901103, "name": "Кофе растворимый", "category": "Напитки", "brand": "Nescafe", "price": 399.90, "unit": "г", "size": 95, "tags": "beverages", "components": ["beverage"]},

    # ========== ПОЛУФАБРИКАТЫ ==========
    {"id": 901201, "name": "Пельмени Сибирские", "category": "Замороженные продукты", "brand": "Сибирская коллекция", "price": 189.90, "unit": "г", "size": 800, "tags": "frozen", "components": ["main_course"]},
    {"id": 901202, "name": "Вареники с картошкой", "category": "Замороженные продукты", "brand": "Сибирская коллекция", "price": 149.90, "unit": "г", "size": 800, "tags": "frozen", "components": ["main_course"]},

    # ========== ДЕШЕВЫЕ АЛЬТЕРНАТИВЫ ==========
    {"id": 901301, "name": "Сосиски молочные", "category": "Колбасные изделия", "brand": "Дымов", "price": 149.90, "unit": "г", "size": 400, "tags": "meat", "components": ["main_course"]},
    {"id": 901302, "name": "Сыр плавленый", "category": "Сыр", "brand": "Дружба", "price": 89.90, "unit": "г", "size": 200, "tags": "dairy", "components": ["snack"]},
    {"id": 901303, "name": "Хлеб батон", "category": "Хлеб", "brand": "Коломенское", "price": 39.90, "unit": "г", "size": 400, "tags": "bakery", "components": ["bakery"]},
]


def add_mock_products():
    """Добавляет mock продукты с embeddings."""
    
    print("=" * 70)
    print("🥗 СОЗДАНИЕ ПОЛНОГО НАБОРА MOCK ПРОДУКТОВ")
    print("=" * 70)
    
    # 1. Загружаем модель
    print("\n🔄 Загрузка модели...")
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    print("   ✅ Модель загружена")
    
    # 2. Подключаемся к БД
    print(f"\n📂 Подключение к {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 3. Удаляем старые mock товары
    print("\n🗑️  Удаление старых mock товаров...")
    cursor.execute("DELETE FROM products WHERE id >= 900000")
    conn.commit()
    print(f"   ✅ Очищено")
    
    # 4. Добавляем новые
    print(f"\n🥦 Добавление {len(MOCK_PRODUCTS)} продуктов...")
    
    for product in MOCK_PRODUCTS:
        # Генерируем embedding
        text = f"{product['name']} {product['category']} {product.get('brand', '')}"
        embedding = model.encode(text, convert_to_numpy=True)
        embedding_blob = embedding.astype(np.float32).tobytes()
        
        cursor.execute("""
            INSERT INTO products 
            (id, product_name, product_category, brand, price_per_unit, unit, 
             package_size, tags, meal_components, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            product['id'],
            product['name'],
            product['category'],
            product['brand'],
            product['price'],
            product['unit'],
            product['size'],
            product['tags'],
            json.dumps(product['components']),
            embedding_blob
        ))
        print(f"   ✅ {product['name']}")
    
    conn.commit()
    conn.close()
    
    print("\n" + "=" * 70)
    print(f"✅ Добавлено {len(MOCK_PRODUCTS)} продуктов")
    print("=" * 70)


if __name__ == '__main__':
    add_mock_products()
