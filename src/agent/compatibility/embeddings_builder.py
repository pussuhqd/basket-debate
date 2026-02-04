"""
Модуль для предвычисления embeddings товаров.

Этот скрипт:
1. Загружает модель SentenceTransformer
2. Читает все товары из products.db
3. Генерирует embeddings для каждого товара
4. Сохраняет embeddings обратно в БД (колонка embedding)

Запуск:
    uv run python src/agent/compatibility/embeddings_builder.py
"""

import sqlite3
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
import torch
from typing import List, Tuple
import sys


# ==================== КОНФИГУРАЦИЯ ====================

# Путь к БД (относительно корня проекта)
DB_PATH = Path("data/processed/products.db")

# Модель для embeddings
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Размер батча для кодирования (зависит от RAM)
BATCH_SIZE = 256  # Для M5 32GB можно и больше, но начнем консервативно

# Устройство (CPU, MPS для M1/M2/M3/M5, или CUDA)
# MPS - Metal Performance Shaders для Apple Silicon
DEVICE = None  # None = auto-detect


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def get_device() -> str:
    """
    Автоматически определяет оптимальное устройство для torch.
    
    Returns:
        str: 'mps', 'cuda', или 'cpu'
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def check_db_exists() -> bool:
    """Проверяет, существует ли БД."""
    if not DB_PATH.exists():
        print(f"❌ Ошибка: БД не найдена по пути {DB_PATH}")
        print(f"   Убедитесь, что вы запустили src/process_dataset.py")
        return False
    return True


def add_embedding_column():
    """
    Добавляет колонку 'embedding' в таблицу products, если её ещё нет.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Проверяем, есть ли уже колонка embedding
    cursor.execute("PRAGMA table_info(products)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if "embedding" not in columns:
        print("📝 Добавляем колонку 'embedding' в таблицу products...")
        cursor.execute("ALTER TABLE products ADD COLUMN embedding BLOB")
        conn.commit()
        print("   ✅ Колонка добавлена")
    else:
        print("   ℹ️  Колонка 'embedding' уже существует")
    
    conn.close()


def load_products() -> List[Tuple[int, str, str]]:
    """
    Загружает все товары из БД.
    
    Returns:
        List[Tuple[int, str, str]]: [(product_id, product_name, product_category), ...]
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Загружаем id, название и категорию
    cursor.execute("""
        SELECT id, product_name, product_category
        FROM products
        ORDER BY id
    """)
    
    products = cursor.fetchall()
    conn.close()
    
    return products


def create_text_for_embedding(product_name: str, product_category: str) -> str:
    """
    Создаёт текст для кодирования в embedding.
    
    Стратегия: Название + Категория
    Пример: "Молоко Правильное 3.2% Молоко"
    
    Args:
        product_name: Название товара
        product_category: Категория товара
    
    Returns:
        str: Текст для embedding
    """
    # Убираем лишние пробелы и None
    name = str(product_name).strip() if product_name else ""
    category = str(product_category).strip() if product_category else ""
    
    # Объединяем название и категорию
    # Категория дублируется для усиления семантики
    text = f"{name} {category}".strip()
    
    return text


def save_embeddings_batch(product_ids: List[int], embeddings: np.ndarray):
    """
    Сохраняет батч embeddings в БД.
    
    Args:
        product_ids: Список ID товаров
        embeddings: Numpy array размером (batch_size, embedding_dim)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Подготавливаем данные для batch update
    data = []
    for product_id, embedding in zip(product_ids, embeddings):
        # Сериализуем numpy array в bytes через pickle
        embedding_bytes = pickle.dumps(embedding, protocol=pickle.HIGHEST_PROTOCOL)
        data.append((embedding_bytes, product_id))
    
    # Batch update
    cursor.executemany("""
        UPDATE products
        SET embedding = ?
        WHERE id = ?
    """, data)
    
    conn.commit()
    conn.close()


# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================

def build_embeddings():
    """
    Главная функция для предвычисления embeddings.
    """
    print("=" * 70)
    print("🚀 ПРЕДВЫЧИСЛЕНИЕ EMBEDDINGS ДЛЯ ТОВАРОВ")
    print("=" * 70)
    
    # 1. Проверяем наличие БД
    if not check_db_exists():
        sys.exit(1)
    
    # 2. Добавляем колонку embedding (если нужно)
    add_embedding_column()
    
    # 3. Определяем устройство
    device = DEVICE if DEVICE else get_device()
    print(f"\n🖥️  Устройство: {device.upper()}")
    
    if device == "mps":
        print("   ℹ️  Используется Metal Performance Shaders (Apple Silicon)")
    elif device == "cuda":
        print(f"   ℹ️  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   ⚠️  Используется CPU (может быть медленно)")
    
    # 4. Загружаем модель
    print(f"\n📦 Загружаем модель: {MODEL_NAME}")
    print("   ⏳ Первая загрузка может занять 1-2 минуты...")
    
    model = SentenceTransformer(MODEL_NAME, device=device)
    embedding_dim = model.get_sentence_embedding_dimension()
    
    print(f"   ✅ Модель загружена")
    print(f"   📊 Размерность embeddings: {embedding_dim}")
    
    # 5. Загружаем товары из БД
    print(f"\n📚 Загружаем товары из БД...")
    products = load_products()
    total_products = len(products)
    
    print(f"   ✅ Загружено {total_products:,} товаров")
    
    # 6. Генерируем embeddings батчами
    print(f"\n🔄 Генерация embeddings (batch_size={BATCH_SIZE})...")
    print("=" * 70)
    
    num_batches = (total_products + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in tqdm(range(num_batches), desc="Обработка батчей"):
        # Берем срез товаров для текущего батча
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_products)
        batch_products = products[start_idx:end_idx]
        
        # Формируем тексты для embedding
        batch_texts = [
            create_text_for_embedding(name, category)
            for _, name, category in batch_products
        ]
        
        # Генерируем embeddings
        # show_progress_bar=False чтобы не дублировать прогресс-бары
        batch_embeddings = model.encode(
            batch_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=BATCH_SIZE
        )
        
        # Сохраняем в БД
        batch_ids = [product_id for product_id, _, _ in batch_products]
        save_embeddings_batch(batch_ids, batch_embeddings)
    
    print("\n" + "=" * 70)
    print("🎉 EMBEDDINGS УСПЕШНО СОЗДАНЫ")
    print("=" * 70)
    print(f"Обработано товаров: {total_products:,}")
    print(f"Размерность embedding: {embedding_dim}")
    print(f"Размер одного embedding: {embedding_dim * 4 / 1024:.2f} KB (float32)")
    print(f"Общий размер embeddings: {total_products * embedding_dim * 4 / 1024 / 1024:.2f} MB")
    print("=" * 70)


# ==================== ТЕСТИРОВАНИЕ ====================

def test_embeddings(num_samples: int = 5):
    """
    Тестирует, что embeddings корректно сохранены.
    
    Args:
        num_samples: Количество товаров для проверки
    """
    print("\n🧪 Тестирование embeddings...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(f"""
        SELECT id, product_name, product_category, embedding
        FROM products
        WHERE embedding IS NOT NULL
        LIMIT {num_samples}
    """)
    
    samples = cursor.fetchall()
    
    if not samples:
        print("   ❌ Не найдено товаров с embeddings!")
        conn.close()
        return
    
    print(f"   ✅ Найдено {len(samples)} товаров с embeddings")
    print("\nПримеры:")
    
    for product_id, name, category, embedding_bytes in samples:
        # Десериализуем embedding
        embedding = pickle.loads(embedding_bytes)
        
        print(f"\n   ID {product_id}: {name}")
        print(f"   Категория: {category}")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Первые 5 значений: {embedding[:5]}")
        print(f"   L2 norm: {np.linalg.norm(embedding):.4f}")
    
    # Проверяем количество товаров с embeddings
    cursor.execute("SELECT COUNT(*) FROM products WHERE embedding IS NOT NULL")
    count_with_embeddings = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM products")
    total_count = cursor.fetchone()[0]
    
    print(f"\n📊 Статистика:")
    print(f"   Всего товаров: {total_count:,}")
    print(f"   С embeddings: {count_with_embeddings:,}")
    print(f"   Без embeddings: {total_count - count_with_embeddings:,}")
    
    if count_with_embeddings == total_count:
        print("   ✅ Все товары обработаны!")
    else:
        print(f"   ⚠️  {total_count - count_with_embeddings} товаров без embeddings")
    
    conn.close()


# ==================== MAIN ====================

if __name__ == "__main__":
    # Генерируем embeddings
    build_embeddings()
    
    # Тестируем результат
    test_embeddings(num_samples=5)
