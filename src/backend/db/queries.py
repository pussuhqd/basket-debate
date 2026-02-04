# src/backend/db/queries.py
import sqlite3
from typing import List, Dict, Optional
from pathlib import Path

DB_PATH = Path(__file__).parents[3] / "data" / "processed" / "products.db"


# ==================== ДОБАВИТЬ В fetch_candidate_products() ====================
def fetch_candidate_products(
    constraints: Dict,
    limit: int = 100,
    max_price_ratio: float = 0.3,
    require_meal_components: bool = False  # ← НОВЫЙ ПАРАМЕТР
) -> List[Dict]:
    """
    Фильтрует товары из БД по constraints и возвращает кандидатов.
    """
    budget = constraints.get("budget_rub") or 5000
    exclude_tags = constraints.get("exclude_tags") or []
    include_tags = constraints.get("include_tags", [])
    people = constraints.get("people", 1)
    
    min_price = budget * 0.02
    max_price = budget * max_price_ratio
    
    query = """
        SELECT id, product_name, product_category, brand, 
            package_size, unit, price_per_unit, tags, meal_components
        FROM products
        WHERE price_per_unit >= ?
        AND price_per_unit <= ?
    """
    params = [min_price, max_price]
    
    # ← НОВОЕ: Фильтрация по meal_components (исключаем 'other')
    if require_meal_components:
        query += " AND meal_components IS NOT NULL AND meal_components != 'other'"
    
    # Фильтрация по exclude_tags
    for tag in exclude_tags:
        query += " AND (tags IS NULL OR tags NOT LIKE ?)"
        params.append(f"%{tag}%")
    
    # Фильтрация по include_tags
    if include_tags:
        for tag in include_tags:
            query += " AND tags LIKE ?"
            params.append(f"%{tag}%")
    
    query += """
        ORDER BY RANDOM()
        LIMIT ?
    """
    params.append(limit)
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
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
            "tags": row["tags"].split("|") if row["tags"] else [],
            "meal_components": row["meal_components"].split("|") if row["meal_components"] else ["other"]  # ← НОВОЕ
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
