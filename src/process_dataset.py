# src/process_dataset.py
import pandas as pd
import sqlite3
from pathlib import Path
from tqdm import tqdm
import re
import math
import json


JSON_PATH = Path("data/tag_rules_extended.json") 
MEAL_PATH = Path("data/meal_components_extended.json") 
INPUT_FILE = Path("data/raw/russian_supermarket_prices.csv")      
DB_PATH = Path("data/processed/products.db")      


with open(JSON_PATH, "r", encoding="utf-8") as f:
    TAG_RULES = json.load(f)

with open(MEAL_PATH, "r", encoding="utf-8") as f:
    MEAL_DATA = json.load(f)


USECOLS = ['product_name', 'product_category', 'brand',
           'package_size', 'unit', 'new_price']


DB_SCHEMA = {
    "product_name": "TEXT",
    "product_category": "TEXT",
    "brand": "TEXT",
    "package_size": "REAL",
    "unit": "TEXT",
    "price_per_unit": "REAL",
    "tags": "TEXT",
    "meal_components": "TEXT"
}


# ==================== НОВОЕ: Списки для фильтрации ====================

# Нерелевантные категории (бытовая химия, корма, косметика)
EXCLUDED_CATEGORIES = [
    'гель для стирки', 'стиральный порошок', 'порошок', 'гель',
    'пятновыводитель', 'средство для мытья посуды', 'моющее средство',
    'бытовая химия', 'корм для кошек', 'корм для собак', 'корм для животных',
    'косметика', 'шампунь', 'бальзам', 'кондиционер для волос',
    'мыло твердое', 'мыло жидкое', 'мыло', 'дезодорант', 'крем',
    'зубная паста', 'зубная щетка', 'бритва', 'туалетная бумага',
    'салфетки', 'подгузники', 'прокладки'
]

MAX_REASONABLE_PRICE = 3000  


def assign_meal_components(product_name, product_category):
    """
    ✅ ИСПРАВЛЕНО: Ограничиваем максимум 2 компонента на товар
    
    Автоматическое присвоение meal_components на основе названия и категории.
    
    Returns:
        List[str]: список компонентов (максимум 2, например, ['main_course', 'side_dish'])
    """
    name = str(product_name).lower()
    category = str(product_category).lower()
    text = f"{name} {category}"
    
    components = set()
    matched_categories = []  # Список совпавших категорий для приоритизации
    
    # Проходим по всем категориям продуктов
    product_categories = MEAL_DATA.get('product_categories', {})
    
    for category_name, category_data in product_categories.items():
        keywords = category_data.get('name', [])
        
        for keyword in keywords:
            if keyword.lower() in text:
                meal_comps = category_data.get('attributes', {}).get('meal_components', [])
                matched_categories.append((category_name, meal_comps))
                components.update(meal_comps)
                break
    
    # Преобразуем set в список
    result = list(components)
    
    if len(result) > 2:
        # Приоритизация компонентов (основные блюда важнее снеков)
        priority_order = [
            'main_course',
            'side_dish',
            'beverage',
            'salad',
            'bakery',
            'sauce',
            'dessert',
            'snack'
        ]
        
        # Сортируем по приоритету
        result_sorted = []
        for comp in priority_order:
            if comp in result:
                result_sorted.append(comp)
        
        # Берём первые 2
        result = result_sorted[:2]
    
    # Если ничего не нашли, возвращаем 'other'
    return result if result else ['other']



def to_float(x):
    try:
        return float(str(x).replace(',', '.'))
    except Exception:
        return math.nan


def create_db_schema():
    """Создаёт таблицы с ТВОЕЙ схемой."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS products")
    columns_sql = ", ".join([f"{k} {v}" for k, v in DB_SCHEMA.items()])
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {columns_sql}
        )
    """)
    conn.commit()
    conn.close()


def normalize_price(price, size, unit):
    """
    Возвращает (нормализованную цену, нормализованный unit).
    
    Нормализация:
    - г → кг (цена за кг)
    - мл → л (цена за литр)
    - шт → шт (цена за штуку)
    - кг → кг (цена за кг)
    - л → л (цена за литр)
    
    Returns:
        tuple: (price_per_unit, normalized_unit)
    """
    if math.isnan(size) or size <= 0:
        return math.nan, None
    
    unit = str(unit).lower().strip()
    
    # 1. Граммы → нормализуем к килограммам
    if unit == 'г':
        price_per_kg = round(price / size * 1000, 2)
        return price_per_kg, 'кг'
    
    # 2. Миллилитры → нормализуем к литрам
    if unit == 'мл':
        price_per_liter = round(price / size * 1000, 2)
        return price_per_liter, 'л'
    
    # 3. Килограммы (уже нормализованы)
    if unit == 'кг':
        price_per_kg = round(price / size, 2)
        return price_per_kg, 'кг'
    
    # 4. Литры (уже нормализованы)
    if unit == 'л':
        price_per_liter = round(price / size, 2)
        return price_per_liter, 'л'
    
    # 5. Штучные товары
    if unit == 'шт':
        price_per_piece = round(price / size, 2)
        return price_per_piece, 'шт'
    
    # 6. Неизвестные единицы → пропускаем
    return math.nan, None


def clean_product_name(name):
    """Убирает размер упаковки из названия (например, '500г', '1л')."""
    pattern = r'\s*\d+[.,]?\d*\s*(?:г|мл|л|кг|шт|уп|упаковка|пачка|бут|банка)\b.*'
    cleaned = re.sub(pattern, '', str(name), flags=re.IGNORECASE).strip()
    return cleaned


def extract_tags(product_name, product_category):
    """Извлекает теги на основе правил из tag_rules.json."""
    name = str(product_name).lower()
    category = str(product_category).lower()
    
    tags = set()
    
    for tag, rules in TAG_RULES.items():
        # Пропускаем вложенные структуры (allergen_markers, quality_markers, certification)
        if not isinstance(rules, dict):
            continue
        
        for field, keywords in rules.items():
            # Пропускаем поля, которые не являются списками
            if not isinstance(keywords, list):
                continue
            
            text = name if field == "name" else category
            
            if any(word in text for word in keywords):
                tags.add(tag)
                break  # Нашли совпадение для этого тега
    
    return sorted(tags)


def is_valid_product(row, price_per_unit, normalized_unit):
    """
    Проверяет, валидный ли товар для включения в БД.
    
    Фильтры:
    1. Нерелевантные категории (корма, бытовая химия)
    2. Аномальные цены (>3000₽/кг или NaN)
    3. Некорректные единицы измерения
    """
    
    # 1. Проверка цены
    if math.isnan(price_per_unit) or price_per_unit <= 0:
        return False, "Некорректная цена"
    
    if price_per_unit > MAX_REASONABLE_PRICE:
        return False, f"Слишком дорого ({price_per_unit:.2f}₽)"
    
    # 2. Проверка единицы измерения
    if normalized_unit not in ['кг', 'л', 'шт']:
        return False, f"Неизвестная единица ({normalized_unit})"
    
    # 3. Проверка категории (исключаем бытовую химию, корма и т.д.)
    category = str(row['product_category']).lower()
    for excluded in EXCLUDED_CATEGORIES:
        if excluded in category:
            return False, f"Исключённая категория ({row['product_category']})"
    
    return True, "OK"


def normalize_row(row):
    """
    Нормализует одну строку датасета.
    
    Returns:
        dict или None (если товар невалидный)
    """
    name = clean_product_name(row['product_name'])
    
    size = to_float(row['package_size'])
    unit = row['unit']
    price = row['new_price']
    
    # Нормализуем цену и единицу измерения
    price_per_unit, normalized_unit = normalize_price(price, size, unit)
    
    # Проверяем валидность товара
    is_valid, reason = is_valid_product(row, price_per_unit, normalized_unit)
    
    if not is_valid:
        return None  # Пропускаем невалидные товары
    
    # Извлекаем теги
    tags = extract_tags(
        product_name=name,
        product_category=row['product_category']
    )
    
    # Извлекаем meal_components (ИСПРАВЛЕНО!)
    meal_components = assign_meal_components(
        product_name=name,
        product_category=row['product_category']
    )
    
    return {
        "product_name": name,
        "product_category": row['product_category'],
        "brand": row['brand'],
        "package_size": size,
        "unit": normalized_unit,
        "price_per_unit": price_per_unit,
        "tags": "|".join(tags),
        "meal_components": "|".join(meal_components)
    }


def process_chunk(chunk):
    """Обрабатывает чанк датасета."""
    chunk = chunk.dropna(subset=['product_name', 'new_price'])
    rows = []
    
    skipped = 0
    
    for _, row in chunk.iterrows():
        normalized = normalize_row(row)
        if normalized is not None:
            rows.append(normalized)
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"  ⚠️  Пропущено {skipped} невалидных товаров")
    
    return pd.DataFrame(rows)


def main():
    """Главная функция обработки датасета."""
    DB_PATH.parent.mkdir(exist_ok=True)
    create_db_schema()
    
    print("=" * 70)
    print("🔄 ОБРАБОТКА ДАТАСЕТА")
    print("=" * 70)
    print(f"Входной файл: {INPUT_FILE}")
    print(f"Выходная БД: {DB_PATH}")
    print(f"Максимальная цена: {MAX_REASONABLE_PRICE}₽/кг")
    print(f"Исключённых категорий: {len(EXCLUDED_CATEGORIES)}")
    print("=" * 70)
    
    total_processed = 0
    total_loaded = 0
    chunksize = 50_000
    
    conn = sqlite3.connect(DB_PATH)
    
    for chunk_num, chunk in enumerate(
        pd.read_csv(INPUT_FILE, usecols=USECOLS, chunksize=chunksize)
    ):
        print(f"\n📦 Чанк {chunk_num + 1}: {len(chunk)} строк")
        total_processed += len(chunk)
        
        processed = process_chunk(chunk)
        
        if not processed.empty:
            processed.to_sql('products', conn, if_exists='append', index=False)
            total_loaded += len(processed)
            print(f"  ✅ Загружено {len(processed)} товаров")
    
    conn.close()
    
    print("\n" + "=" * 70)
    print("🎉 ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 70)
    print(f"Обработано строк: {total_processed}")
    print(f"Загружено товаров: {total_loaded}")
    print(f"Отфильтровано: {total_processed - total_loaded} ({(total_processed - total_loaded) / total_processed * 100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
