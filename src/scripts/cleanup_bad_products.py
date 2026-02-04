"""
Удаляет проблемные товары из БД.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path('data/processed/products.db')

# Товары которые мешают поиску
BAD_KEYWORDS = [
    'палтус',
    'конфет',
    'шоколад',
    'чипс',
    'снек',
    'корм для',
    'мыло',
    'шампунь',
    'бытовая химия',
    'стиральный',
    'освежитель',
    'салфетки',
    'игрушк',
    'детское питание',
    'пюре "фруто"',
    'нектар "фруто"',
]

def cleanup():
    """Удаляет мусорные товары."""
    
    print("=" * 70)
    print("🗑️  ОЧИСТКА БД ОТ ПРОБЛЕМНЫХ ТОВАРОВ")
    print("=" * 70)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Считаем товары до очистки
    cursor.execute("SELECT COUNT(*) FROM products WHERE id < 900000")
    before = cursor.fetchone()[0]
    
    print(f"\n📊 Товаров до очистки: {before}")
    
    # Удаляем по каждому ключевому слову
    deleted_total = 0
    
    for keyword in BAD_KEYWORDS:
        cursor.execute(f"""
            DELETE FROM products 
            WHERE id < 900000 
            AND (product_name LIKE '%{keyword}%' OR product_category LIKE '%{keyword}%')
        """)
        deleted = cursor.rowcount
        if deleted > 0:
            print(f"   ❌ Удалено по '{keyword}': {deleted}")
            deleted_total += deleted
    
    # Удаляем конкретный палтус
    cursor.execute("DELETE FROM products WHERE product_name LIKE '%Палтус%'")
    deleted_total += cursor.rowcount
    
    conn.commit()
    
    # Считаем после очистки
    cursor.execute("SELECT COUNT(*) FROM products WHERE id < 900000")
    after = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM products WHERE id >= 900000")
    mock_count = cursor.fetchone()[0]
    
    conn.close()
    
    print("\n" + "=" * 70)
    print(f"✅ Удалено: {deleted_total} товаров")
    print(f"📊 Осталось: {after} (оригинальных) + {mock_count} (mock)")
    print("=" * 70)


if __name__ == '__main__':
    cleanup()
