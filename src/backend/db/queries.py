# src/backend/db/queries.py
import sqlite3
from typing import List, Dict, Optional
from pathlib import Path

DB_PATH = Path(__file__).parents[3] / "data" / "processed" / "products.db"


def fetch_candidate_products(
    constraints: Dict,
    limit: int = 100,
    max_price_ratio: float = 0.3  # Максимальная цена товара = 30% от бюджета
) -> List[Dict]:
    """
    Фильтрует товары из БД по constraints и возвращает кандидатов.
    
    Args:
        constraints: словарь с ключами:
            - budget_rub (float): бюджет в рублях
            - exclude_tags (List[str]): теги для исключения
            - include_tags (List[str]): обязательные теги
            - meal_type (List[str]): тип приёма пищи (не используется пока)
            - people (int): количество людей
        limit: максимальное количество товаров
        max_price_ratio: максимальная цена товара относительно бюджета
    
    Returns:
        List[Dict]: список товаров с полями id, product_name, price_per_unit, tags
    """
    
    budget = constraints.get("budget_rub") or 5000  
    exclude_tags = constraints.get("exclude_tags") or []  
    include_tags = constraints.get("include_tags", [])
    people = constraints.get("people", 1)
    
    # Расчёт максимальной цены за единицу (чтобы не брать слишком дорогие товары)
    min_price = budget * 0.02
    max_price = budget * max_price_ratio
    
    # Строим SQL-запрос
    query = """
        SELECT id, product_name, product_category, brand, 
            package_size, unit, price_per_unit, tags
        FROM products
        WHERE price_per_unit >= ?  
        AND price_per_unit <= ?
    """
    params = [min_price, max_price]  # НОВОЕ: два параметра
    
    # Фильтрация по exclude_tags (исключаем товары с этими тегами)
    for tag in exclude_tags:
        query += " AND (tags IS NULL OR tags NOT LIKE ?)"
        params.append(f"%{tag}%")
    
    # Фильтрация по include_tags (берём только товары с этими тегами)
    if include_tags:
        for tag in include_tags:
            query += " AND tags LIKE ?"
            params.append(f"%{tag}%")
    
    # Сортировка: сначала дешёвые, потом разнообразие по категориям
    query += """
        ORDER BY price_per_unit ASC
        LIMIT ?
    """
    params.append(limit)
    
    # Выполняем запрос
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Возвращать строки как словари
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    # Преобразуем в список словарей
    products = []
    for row in rows:
        products.append({
            "id": row["id"],
            "product_name": row["product_name"],
            "product_category": row["product_category"],
            "brand": row["brand"],
            "package_size": row["package_size"],
            "unit": row["unit"],
            "price_per_unit": row["price_per_unit"],
            "tags": row["tags"].split("|") if row["tags"] else []
        })
    
    return products


if __name__ == "__main__":
    test_constraints = {
        "budget_rub": 1500,
        "exclude_tags": ["dairy"],
        "include_tags": [""],
        "people": 2
    }
    products = fetch_candidate_products(test_constraints, limit=20)
    print(f"Найдено {len(products)} товаров:")
    for p in products[:5]:
        print(f"  - {p['product_name']}: {p['price_per_unit']}₽/{p['unit']}, теги: {p['tags']}\n")
