import sqlite3
import numpy as np
import pickle

conn = sqlite3.connect('data/processed/products.db')
c = conn.cursor()

# Проверяем старые товары
c.execute("SELECT id, product_name, embedding FROM products WHERE id < 900000 AND embedding IS NOT NULL LIMIT 5")
old_products = c.fetchall()

# Проверяем mock товары
c.execute("SELECT id, product_name, embedding FROM products WHERE id >= 900000 AND embedding IS NOT NULL LIMIT 5")
mock_products = c.fetchall()

print("=" * 70)
print("🔍 ДИАГНОСТИКА EMBEDDINGS")
print("=" * 70)

# Старые товары
print("\n📦 СТАРЫЕ ТОВАРЫ:")
for pid, name, emb_blob in old_products:
    # Пробуем разные форматы
    format_type = "?"
    size = len(emb_blob) if emb_blob else 0
    
    try:
        # Пробуем tobytes
        arr = np.frombuffer(emb_blob, dtype=np.float32)
        if len(arr) > 0 and np.isfinite(arr).all():
            format_type = "tobytes ✅"
        else:
            format_type = "tobytes (битый)"
    except:
        try:
            # Пробуем pickle
            arr = pickle.loads(emb_blob)
            format_type = "pickle"
        except:
            format_type = "НЕИЗВЕСТНЫЙ ❌"
    
    print(f"   {pid}: {name[:40]} - {format_type} ({size} bytes)")

# Mock товары
print("\n🆕 MOCK ТОВАРЫ:")
for pid, name, emb_blob in mock_products:
    format_type = "?"
    size = len(emb_blob) if emb_blob else 0
    
    try:
        arr = np.frombuffer(emb_blob, dtype=np.float32)
        if len(arr) > 0 and np.isfinite(arr).all():
            format_type = "tobytes ✅"
        else:
            format_type = "tobytes (битый)"
    except:
        try:
            arr = pickle.loads(emb_blob)
            format_type = "pickle"
        except:
            format_type = "НЕИЗВЕСТНЫЙ ❌"
    
    print(f"   {pid}: {name[:40]} - {format_type} ({size} bytes)")

# Статистика
c.execute("SELECT COUNT(*) FROM products WHERE id < 900000 AND embedding IS NOT NULL")
old_count = c.fetchone()[0]

c.execute("SELECT COUNT(*) FROM products WHERE id >= 900000 AND embedding IS NOT NULL")
mock_count = c.fetchone()[0]

c.execute("SELECT COUNT(*) FROM products WHERE id < 900000")
old_total = c.fetchone()[0]

print(f"\n📊 СТАТИСТИКА:")
print(f"   Старых товаров: {old_total}")
print(f"   - С embeddings: {old_count}")
print(f"   - Без embeddings: {old_total - old_count}")
print(f"   Mock товаров с embeddings: {mock_count}")

conn.close()

print("\n" + "=" * 70)
