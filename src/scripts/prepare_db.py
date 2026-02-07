# src/scripts/prepare_db.py
"""
Единый пайплайн подготовки базы данных products.db.

Этапы:
1. Обработка CSV → SQLite (process_dataset)
2. Очистка от мусорных товаров (cleanup)
3. Добавление mock товаров (add_mocks)

Примечание: Embeddings генерируются отдельно через build_embeddings.py

Запуск:
    # Полный пайплайн
    uv run python -m src.scripts.prepare_db
    
    # Только этап 1 (обработка CSV)
    uv run python -m src.scripts.prepare_db --step process
    
    # Пропустить mock товары
    uv run python -m src.scripts.prepare_db --no-mocks
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# ==================== ИМПОРТЫ ====================
from src.utils.queries import get_connection, DB_PATH


# ==================== КОНФИГУРАЦИЯ ====================

PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "raw" / "russian_supermarket_prices.csv"
TAG_RULES_PATH = PROJECT_ROOT / "data" / "templates" / "tag_rules_extended.json"
MOCK_PRODUCTS_PATH = PROJECT_ROOT / "data" / "templates" / "mock.json"
MEAL_COMPONENTS_PATH = PROJECT_ROOT / "data" / "templates" /"meal_components_extended.json"

CHUNKSIZE = 50_000
MAX_REASONABLE_PRICE = 3000  # ₽/кг
USECOLS = ['product_name', 'product_category', 'brand', 'package_size', 'unit', 'new_price']

# Исключённые категории
EXCLUDED_CATEGORIES = [
    'гель для стирки', 'стиральный порошок', 'порошок', 'гель',
    'пятновыводитель', 'средство для мытья посуды', 'моющее средство',
    'бытовая химия', 'корм для кошек', 'корм для собак', 'корм для животных',
    'косметика', 'шампунь', 'бальзам', 'кондиционер для волос',
    'мыло твердое', 'мыло жидкое', 'мыло', 'дезодорант', 'крем',
    'зубная паста', 'зубная щетка', 'бритва', 'туалетная бумага',
    'салфетки', 'подгузники', 'прокладки'
]

# Плохие ключевые слова
BAD_KEYWORDS = [
    'палтус', 'конфет', 'шоколад', 'чипс', 'снек', 'корм для',
    'мыло', 'шампунь', 'бытовая химия', 'стиральный', 'освежитель',
    'салфетки', 'игрушк', 'детское питание', 'пюре "фруто"', 'нектар "фруто"'
]


# ==================== ЗАГРУЗКА ПРАВИЛ ====================

def load_rules():
    """Загружает правила тегов и meal_components."""
    with open(TAG_RULES_PATH, 'r', encoding='utf-8') as f:
        tag_rules = json.load(f)
    
    with open(MEAL_COMPONENTS_PATH, 'r', encoding='utf-8') as f:
        meal_data = json.load(f)
    
    with open(MOCK_PRODUCTS_PATH, 'r', encoding='utf-8') as f:
        mock_data = json.load(f)

    return tag_rules, meal_data, mock_data


TAG_RULES, MEAL_DATA, MOCK_PRODUCTS = load_rules()


# ==================== ЭТАП 1: ОБРАБОТКА CSV ====================

def create_db_schema():
    """Создаёт пустую таблицу products."""
    conn = get_connection()
    conn.execute("DROP TABLE IF EXISTS products")
    conn.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_name TEXT,
            product_category TEXT,
            brand TEXT,
            package_size REAL,
            unit TEXT,
            price_per_unit REAL,
            tags TEXT,
            meal_components TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()
    print("   ✅ Таблица products создана")


def clean_product_name(name: str) -> str:
    """Убирает размер упаковки из названия."""
    pattern = r'\s*\d+[.,]?\d*\s*(?:г|мл|л|кг|шт|уп|упаковка|пачка|бут|банка)\b.*'
    cleaned = re.sub(pattern, '', str(name), flags=re.IGNORECASE).strip()
    return cleaned


def to_float(x) -> float:
    """Конвертирует строку в float."""
    try:
        return float(str(x).replace(',', '.'))
    except Exception:
        return math.nan


def normalize_price(price: float, size: float, unit: str) -> Tuple[float, float, Optional[str]]:
    """
    Нормализует цену И размер упаковки к базовым единицам (кг, л, шт).
    
    Returns:
        (price_per_unit, normalized_size, normalized_unit)
    """
    if math.isnan(size) or size <= 0:
        return math.nan, math.nan, None
    
    unit = str(unit).lower().strip()
    price*=1.8
    if unit == 'г':
        return round(price / size * 1000, 2), round(size / 1000, 3), 'кг'
    elif unit == 'мл':
        return round(price / size * 1000, 2), round(size / 1000, 3), 'л'
    elif unit == 'кг':
        return round(price / size, 2), round(size, 3), 'кг'
    elif unit == 'л':
        return round(price / size, 2), round(size, 3), 'л'
    elif unit == 'шт':
        return round(price / size, 2), round(size, 3), 'шт'
    else:
        return math.nan, math.nan, None



def extract_tags(product_name: str, product_category: str) -> List[str]:
    """Извлекает теги на основе tag_rules.json."""
    name = str(product_name).lower()
    category = str(product_category).lower()
    
    tags = set()
    
    for tag, rules in TAG_RULES.items():
        if not isinstance(rules, dict):
            continue
        
        for field, keywords in rules.items():
            if not isinstance(keywords, list):
                continue
            
            text = name if field == "name" else category
            
            if any(word in text for word in keywords):
                tags.add(tag)
                break
    
    return sorted(tags)


def assign_meal_components(product_name: str, product_category: str) -> List[str]:
    """Присваивает meal_components (максимум 2)."""
    name = str(product_name).lower()
    category = str(product_category).lower()
    text = f"{name} {category}"
    
    components = set()
    product_categories = MEAL_DATA.get('product_categories', {})
    
    for category_name, category_data in product_categories.items():
        keywords = category_data.get('name', [])
        
        for keyword in keywords:
            if keyword.lower() in text:
                meal_comps = category_data.get('attributes', {}).get('meal_components', [])
                components.update(meal_comps)
                break
    
    result = list(components)
    
    # Ограничиваем до 2 компонентов
    if len(result) > 2:
        priority_order = [
            'main_course', 'side_dish', 'beverage', 'salad',
            'bakery', 'sauce', 'dessert', 'snack'
        ]
        
        result_sorted = [comp for comp in priority_order if comp in result]
        result = result_sorted[:2]
    
    return result if result else ['other']


def is_valid_product(row, price_per_unit: float, normalized_unit: str) -> Tuple[bool, str]:
    """Проверяет валидность товара."""
    
    # Проверка цены
    if math.isnan(price_per_unit) or price_per_unit <= 0:
        return False, "Некорректная цена"
    
    if price_per_unit > MAX_REASONABLE_PRICE:
        return False, f"Слишком дорого"
    
    # Проверка единицы измерения
    if normalized_unit not in ['кг', 'л', 'шт']:
        return False, f"Неизвестная единица"
    
    # Проверка категории
    category = str(row['product_category']).lower()
    for excluded in EXCLUDED_CATEGORIES:
        if excluded in category:
            return False, "Исключённая категория"
    
    return True, "OK"


def normalize_row(row) -> Optional[Dict]:
    """Нормализует одну строку датасета."""
    name = clean_product_name(row['product_name'])
    size = to_float(row['package_size'])
    unit = row['unit']
    price = row['new_price']
    
    price_per_unit, normalized_size, normalized_unit = normalize_price(price, size, unit)
    is_valid, reason = is_valid_product(row, price_per_unit, normalized_unit)
    
    if not is_valid:
        return None
    
    tags = extract_tags(name, row['product_category'])
    meal_components = assign_meal_components(name, row['product_category'])
    
    return {
        "product_name": name,
        "product_category": row['product_category'],
        "brand": row['brand'],
        "package_size": normalized_size,  # ✅ Теперь в кг/л/шт
        "unit": normalized_unit,
        "price_per_unit": price_per_unit,
        "tags": "|".join(tags),
        "meal_components": "|".join(meal_components)
    }



def process_csv():
    """Обрабатывает CSV и загружает в БД."""
    print("\n" + "=" * 70)
    print("📊 ЭТАП 1: ОБРАБОТКА CSV")
    print("=" * 70)
    
    if not INPUT_CSV.exists():
        print(f"❌ CSV не найден: {INPUT_CSV}")
        return False
    
    create_db_schema()
    
    print(f"Входной файл: {INPUT_CSV}")
    print(f"Макс. цена: {MAX_REASONABLE_PRICE}₽/кг")
    
    total_processed = 0
    total_loaded = 0
    conn = get_connection()
    
    for chunk_num, chunk in enumerate(pd.read_csv(INPUT_CSV, usecols=USECOLS, chunksize=CHUNKSIZE)):
        print(f"\n📦 Чанк {chunk_num + 1}: {len(chunk)} строк")
        total_processed += len(chunk)
        
        chunk = chunk.dropna(subset=['product_name', 'new_price'])
        rows = []
        
        for _, row in chunk.iterrows():
            normalized = normalize_row(row)
            if normalized:
                rows.append(normalized)
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_sql('products', conn, if_exists='append', index=False)
            total_loaded += len(rows)
            print(f"   ✅ Загружено: {len(rows)}")
    
    conn.close()
    
    print(f"\n✅ Загружено: {total_loaded:,} товаров")
    print(f"⚠️  Отфильтровано: {total_processed - total_loaded:,}")
    
    return True


# ==================== ЭТАП 2: ОЧИСТКА ====================

def cleanup_bad_products():
    """Удаляет мусорные товары."""
    print("\n" + "=" * 70)
    print("🗑️  ЭТАП 2: ОЧИСТКА ОТ МУСОРА")
    print("=" * 70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM products WHERE id < 900000")
    before = cursor.fetchone()[0]
    
    print(f"Товаров до очистки: {before:,}")
    
    deleted_total = 0
    
    for keyword in BAD_KEYWORDS:
        cursor.execute(f"""
            DELETE FROM products
            WHERE id < 900000
            AND (product_name LIKE '%{keyword}%' OR product_category LIKE '%{keyword}%')
        """)
        deleted = cursor.rowcount
        if deleted > 0:
            print(f"   ❌ '{keyword}': {deleted}")
            deleted_total += deleted
    
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM products WHERE id < 900000")
    after = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"\n✅ Удалено: {deleted_total} товаров")
    print(f"📊 Осталось: {after:,} товаров")
    
    return True


# ==================== ЭТАП 3: MOCK ТОВАРЫ (БЕЗ EMBEDDINGS) ====================


def add_mock_products():
    """Добавляет mock товары БЕЗ embeddings."""
    print("\n" + "=" * 70)
    print("🥗 ЭТАП 3: ДОБАВЛЕНИЕ MOCK ТОВАРОВ")
    print("=" * 70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Удаляем старые mock
    cursor.execute("DELETE FROM products WHERE id >= 900000")
    conn.commit()
    print("   🗑️  Старые mock удалены")
    
    print(f"\n🥦 Добавление {len(MOCK_PRODUCTS)} товаров...")
    
    for product in tqdm(MOCK_PRODUCTS, desc="Mock товары"):
        cursor.execute("""
            INSERT INTO products
            (id, product_name, product_category, brand, price_per_unit, unit,
             package_size, tags, meal_components)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            product['id'], product['name'], product['category'], product['brand'],
            product['price'], product['unit'], product['size'],
            product['tags'], "|".join(product['components'])
        ))
    
    conn.commit()
    conn.close()
    
    print(f"✅ Добавлено: {len(MOCK_PRODUCTS)} товаров (без embeddings)")
    print(f"ℹ️  Для генерации embeddings запустите: uv run python -m src.scripts.build_embeddings")
    
    return True


# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================

def main():
    """Главная функция пайплайна."""
    parser = argparse.ArgumentParser(description='Подготовка БД products.db')
    parser.add_argument(
        '--step',
        choices=['process', 'cleanup', 'mocks', 'all'],
        default='all',
        help='Выполнить конкретный этап'
    )
    parser.add_argument(
        '--no-mocks',
        action='store_true',
        help='Пропустить добавление mock товаров'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🚀 ПАЙПЛАЙН ПОДГОТОВКИ БАЗЫ ДАННЫХ")
    print("=" * 70)
    print(f"База данных: {DB_PATH}")
    print("=" * 70)
    
    success = True
    
    # Этап 1
    if args.step in ['process', 'all']:
        success = process_csv()
        if not success:
            return
    
    # Этап 2
    if args.step in ['cleanup', 'all']:
        success = cleanup_bad_products()
        if not success:
            return
    
    # Этап 3
    if args.step in ['mocks', 'all'] and not args.no_mocks:
        success = add_mock_products()
        if not success:
            return
    
    # Финальная статистика
    print("\n" + "=" * 70)
    print("📊 ФИНАЛЬНАЯ СТАТИСТИКА")
    print("=" * 70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM products WHERE id < 900000")
    real_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM products WHERE id >= 900000")
    mock_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM products WHERE embedding IS NOT NULL")
    with_embeddings = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"Реальных товаров: {real_count:,}")
    print(f"Mock товаров: {mock_count}")
    print(f"С embeddings: {with_embeddings:,}")
    print(f"Без embeddings: {(real_count + mock_count) - with_embeddings:,}")
    print("=" * 70)
    print("✅ ПАЙПЛАЙН ЗАВЕРШЁН")
    
    if with_embeddings == 0:
        print("\n⚠️  СЛЕДУЮЩИЙ ШАГ:")
        print("   uv run python -m src.scripts.build_embeddings")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
