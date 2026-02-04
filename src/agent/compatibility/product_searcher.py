"""
Модуль для семантического поиска товаров через embeddings.

Основные функции:
- Поиск товаров по текстовому запросу (cosine similarity)
- Фильтрация по meal_components, категориям, тегам
- Ранжирование результатов

Использование:
    searcher = ProductSearcher()
    results = searcher.search(
        query="курица филе",
        meal_component="main_course",
        limit=5
    )
"""

import sqlite3
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
import torch


# ==================== КОНФИГУРАЦИЯ ====================

DB_PATH = Path("data/processed/products.db")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ==================== КЛАСС ProductSearcher ====================

class ProductSearcher:
    """
    Класс для семантического поиска товаров в БД через embeddings.
    """
    
    def __init__(self, db_path: Path = DB_PATH, model_name: str = MODEL_NAME):
        """
        Инициализация поисковика.
        
        Args:
            db_path: Путь к базе данных
            model_name: Название модели SentenceTransformer
        """
        self.db_path = db_path
        self.model_name = model_name
        
        # Определяем устройство
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        # Загружаем модель (кешируется после первой загрузки)
        print(f"🔄 Загрузка модели {model_name} на {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        print("   ✅ Модель загружена")
    
    
    def _load_products_with_embeddings(
        self,
        meal_component: Optional[str] = None,
        category: Optional[str] = None,
        exclude_tags: Optional[List[str]] = None,
        include_tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Загружает товары из БД с фильтрацией.
        
        Args:
            meal_component: Фильтр по meal_component (например, "main_course")
            category: Фильтр по категории (например, "Мясо")
            exclude_tags: Теги для исключения (например, ["dairy"])
            include_tags: Обязательные теги (например, ["vegan"])
        
        Returns:
            List[Dict]: Список товаров с embeddings
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Базовый запрос
        query = """
            SELECT id, product_name, product_category, brand,
                   package_size, unit, price_per_unit,
                   tags, meal_components, embedding
            FROM products
            WHERE embedding IS NOT NULL
        """
        params = []
        
        # Фильтр по meal_component
        if meal_component:
            query += " AND (meal_components LIKE ? OR meal_components LIKE ? OR meal_components LIKE ?)"
            params.extend([
                f"%{meal_component}%",
                f"{meal_component}|%",
                f"%|{meal_component}"
            ])
        
        # Фильтр по категории
        if category:
            query += " AND product_category LIKE ?"
            params.append(f"%{category}%")
        
        # Фильтр по exclude_tags
        if exclude_tags:
            for tag in exclude_tags:
                query += " AND (tags IS NULL OR tags NOT LIKE ?)"
                params.append(f"%{tag}%")
        
        # Фильтр по include_tags
        if include_tags:
            for tag in include_tags:
                query += " AND tags LIKE ?"
                params.append(f"%{tag}%")
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Преобразуем в список словарей
        products = []
        for row in rows:
            # Десериализуем embedding
            embedding = np.frombuffer(row["embedding"], dtype=np.float32)
            
            products.append({
                "id": row["id"],
                "product_name": row["product_name"],
                "product_category": row["product_category"],
                "brand": row["brand"],
                "package_size": row["package_size"],
                "unit": row["unit"],
                "price_per_unit": row["price_per_unit"],
                "tags": row["tags"].split("|") if row["tags"] else [],
                "meal_components": row["meal_components"].split("|") if row["meal_components"] else [],
                "embedding": embedding
            })
        
        return products
    
    
    def search(
        self,
        query: str,
        meal_component: Optional[str] = None,
        category: Optional[str] = None,
        exclude_tags: Optional[List[str]] = None,
        include_tags: Optional[List[str]] = None,
        limit: int = 10,
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Семантический поиск товаров по текстовому запросу.
        
        Args:
            query: Поисковый запрос (например, "курица филе грудка")
            meal_component: Фильтр по meal_component
            category: Фильтр по категории
            exclude_tags: Теги для исключения
            include_tags: Обязательные теги
            limit: Максимальное количество результатов
            min_score: Минимальный score (cosine similarity)
        
        Returns:
            List[Dict]: Список товаров, отсортированных по релевантности
        """
        # 1. Кодируем запрос в embedding
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Нормализуем query embedding для cosine similarity
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # 2. Загружаем товары с фильтрацией
        products = self._load_products_with_embeddings(
            meal_component=meal_component,
            category=category,
            exclude_tags=exclude_tags,
            include_tags=include_tags
        )
        
        if not products:
            return []
        
        # 3. Вычисляем cosine similarity
        product_embeddings = np.array([p["embedding"] for p in products])
        
        # Нормализуем product embeddings
        product_embeddings = product_embeddings / np.linalg.norm(
            product_embeddings, axis=1, keepdims=True
        )
        
        # Считаем cosine similarity (dot product нормализованных векторов)
        similarities = np.dot(product_embeddings, query_embedding)
        
        # 4. Добавляем scores к товарам
        for i, product in enumerate(products):
            product["search_score"] = float(similarities[i])
        
        # 5. Фильтруем по min_score
        products = [p for p in products if p["search_score"] >= min_score]
        
        # 6. Сортируем по убыванию score
        products.sort(key=lambda x: x["search_score"], reverse=True)
        
        # 7. Возвращаем top-N
        return products[:limit]
    
    
    def search_by_ingredient(
        self,
        ingredient_name: str,
        quantity_grams: float,
        meal_component: Optional[str] = None,
        people: int = 1
    ) -> Optional[Dict]:
        """
        Поиск товара для конкретного ингредиента из сценария.
        
        Возвращает лучший товар с расчетом необходимого количества.
        
        Args:
            ingredient_name: Название ингредиента (например, "курица")
            quantity_grams: Количество на 1 человека (в граммах или мл)
            meal_component: Тип компонента (main_course, side_dish и т.д.)
            people: Количество человек
        
        Returns:
            Dict: Товар с полями quantity_needed, total_price
        """
        # Ищем товары
        results = self.search(
            query=ingredient_name,
            meal_component=meal_component,
            limit=5
        )
        
        if not results:
            return None
        
        # Выбираем лучший товар (первый в списке)
        best_product = results[0]
        
        # Рассчитываем необходимое количество
        total_quantity_needed = quantity_grams * people
        
        # Конвертируем в упаковки товара
        package_size = best_product["package_size"]
        
        # Если единица измерения разная (г vs кг), нормализуем
        if best_product["unit"] == "кг":
            package_size_grams = package_size * 1000
        elif best_product["unit"] == "л":
            package_size_grams = package_size * 1000  # Предполагаем 1л = 1кг
        elif best_product["unit"] == "г" or best_product["unit"] == "мл":
            package_size_grams = package_size
        else:  # шт
            package_size_grams = package_size  # Для штучных товаров
        
        # Количество упаковок (округляем вверх)
        num_packages = int(np.ceil(total_quantity_needed / package_size_grams))
        
        # Итоговая цена
        total_price = num_packages * best_product["price_per_unit"]
        
        fractional_cost = (total_quantity_needed / package_size_grams) * best_product["price_per_unit"]

        best_product["fractional_cost"] = round(fractional_cost, 2)  # Реальная стоимость нужного количества
        best_product["full_package_cost"] = round(total_price, 2)    # Стоимость целых упаковок
        best_product["quantity_needed"] = num_packages
        best_product["total_price"] = round(total_price, 2)
        best_product["quantity_grams_per_person"] = quantity_grams
        
        return best_product


# ==================== ТЕСТИРОВАНИЕ ====================

def test_searcher():
    """
    Тестирует работу ProductSearcher.
    """
    print("=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ ProductSearcher")
    print("=" * 70)
    
    searcher = ProductSearcher()
    
    # Тест 1: Поиск курицы
    print("\n📝 Тест 1: Поиск 'курица филе грудка'")
    results = searcher.search(
        query="курица филе грудка",
        meal_component="main_course",
        limit=5
    )
    
    print(f"   Найдено: {len(results)} товаров\n")
    for i, product in enumerate(results, 1):
        print(f"   {i}. {product['product_name']}")
        print(f"      Категория: {product['product_category']}")
        print(f"      Цена: {product['price_per_unit']}₽/{product['unit']}")
        print(f"      Score: {product['search_score']:.4f}")
        print(f"      Components: {', '.join(product['meal_components'])}\n")
    
    # Тест 2: Поиск картофеля
    print("\n📝 Тест 2: Поиск 'картофель'")
    results = searcher.search(
        query="картофель",
        meal_component="side_dish",
        limit=3
    )
    
    print(f"   Найдено: {len(results)} товаров\n")
    for i, product in enumerate(results, 1):
        print(f"   {i}. {product['product_name']}")
        print(f"      Цена: {product['price_per_unit']}₽/{product['unit']}")
        print(f"      Score: {product['search_score']:.4f}\n")
    
    # Тест 3: Поиск с расчетом количества
    print("\n📝 Тест 3: Поиск ингредиента для 3 человек")
    result = searcher.search_by_ingredient(
        ingredient_name="курица",
        quantity_grams=250,  # 250г на человека
        meal_component="main_course",
        people=3
    )
    
    if result:
        print(f"   Товар: {result['product_name']}")
        print(f"   Цена за единицу: {result['price_per_unit']}₽/{result['unit']}")
        print(f"   Размер упаковки: {result['package_size']}{result['unit']}")
        print(f"   Нужно упаковок: {result['quantity_needed']}")
        print(f"   Итоговая цена: {result['total_price']}₽")
        print(f"   Score: {result['search_score']:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ Тестирование завершено")
    print("=" * 70)


if __name__ == "__main__":
    test_searcher()
