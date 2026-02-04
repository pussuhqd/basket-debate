"""
BudgetAgent - оптимизация корзины под бюджет с embeddings.

Работает в отдельном потоке (thread-safe SQLite).
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity


DB_PATH = Path("data/processed/products.db")


class BudgetAgent:
    """
    Агент для оптимизации корзины под бюджет.
    Ищет дешёвые аналоги дорогих товаров используя embeddings.
    """
    
    def __init__(self, db_path: Path = DB_PATH):
        """
        Инициализация агента.
        
        Args:
            db_path: Путь к БД с товарами
        """
        self.db_path = db_path
        print("💰 BudgetAgent инициализирован")
    
    
    def optimize(
        self,
        basket: List[Dict],
        budget_rub: Optional[float] = None,
        min_discount: float = 0.3
    ) -> Dict:
        """
        Оптимизирует корзину под бюджет.
        
        Args:
            basket: Корзина от CompatibilityAgent
            budget_rub: Бюджет в рублях
            min_discount: Минимальная экономия (0.3 = 30%)
        
        Returns:
            Dict: Результат оптимизации
        """
        if not basket:
            return {
                "basket": [],
                "total_price": 0.0,
                "saved": 0.0,
                "replacements": [],
                "within_budget": True,
                "message": "Пустая корзина"
            }
        
        # Считаем текущую цену
        original_price = sum(item.get('price', 0) for item in basket)
        
        # Если бюджет не указан или укладываемся - ничего не делаем
        if budget_rub is None or original_price <= budget_rub:
            return {
                "basket": basket,
                "total_price": original_price,
                "saved": 0.0,
                "replacements": [],
                "within_budget": True,
                "message": "В пределах бюджета"
            }
        
        print(f"\n💰 BudgetAgent: Бюджет превышен на {original_price - budget_rub:.2f}₽")
        print(f"   Ищу дешёвые аналоги...")
        
        # Создаём connection (thread-safe)
        conn = sqlite3.connect(self.db_path)
        
        optimized_basket = basket.copy()
        replacements = []
        total_saved = 0.0
        
        # Сортируем по цене (самые дорогие вверху)
        sorted_indices = sorted(
            range(len(optimized_basket)),
            key=lambda i: optimized_basket[i].get('price', 0),
            reverse=True
        )
        
        # Пытаемся заменить дорогие товары
        for idx in sorted_indices:
            current_price = sum(p.get('price', 0) for p in optimized_basket)
            
            # Если уже уложились - останавливаемся
            if current_price <= budget_rub:
                break
            
            item = optimized_basket[idx]
            
            # Ищем дешёвый аналог
            alternative = self._find_cheaper_alternative(
                item,
                min_discount=min_discount,
                conn=conn
            )
            
            if alternative:
                old_price = item.get('price', 0)
                new_price = alternative.get('price', 0)
                saved = old_price - new_price
                
                # Заменяем товар
                optimized_basket[idx] = alternative
                
                replacements.append({
                    'from': item.get('name', item.get('product_name', '')),
                    'to': alternative.get('name', alternative.get('product_name', '')),
                    'saved': saved
                })
                
                total_saved += saved
                
                print(f"   ✅ {item.get('name', '')[:40]} ({old_price:.2f}₽)")
                print(f"      → {alternative.get('name', '')[:40]} ({new_price:.2f}₽)")
                print(f"      Экономия: {saved:.2f}₽")
        
        # Закрываем connection
        conn.close()
        
        # Итоговая цена
        final_price = sum(p.get('price', 0) for p in optimized_basket)
        
        return {
            "basket": optimized_basket,
            "total_price": final_price,
            "saved": total_saved,
            "replacements": replacements,
            "within_budget": final_price <= budget_rub,
            "message": f"Заменено {len(replacements)} товаров, сэкономлено {total_saved:.2f}₽"
        }
    
    
    def _find_cheaper_alternative(
        self,
        item: Dict,
        min_discount: float = 0.3,
        conn: Optional[sqlite3.Connection] = None
    ) -> Optional[Dict]:
        """
        Ищет дешёвый аналог товара используя embeddings.
        
        Args:
            item: Исходный товар
            min_discount: Минимальная экономия
            conn: SQLite connection (thread-safe)
        
        Returns:
            Dict: Дешёвый аналог или None
        """
        original_price = item.get('price', 0)
        original_embedding = item.get('embedding')
        meal_components = item.get('meal_components', [])
        
        if original_embedding is None:
            return None
        
        # Максимальная цена аналога
        max_price = original_price * (1 - min_discount)
        
        # Создаём connection если не передана
        if conn is None:
            conn = sqlite3.connect(self.db_path)
            close_conn = True
        else:
            close_conn = False
        
        cursor = conn.cursor()
        
        # Ищем похожие товары дешевле
        query = """
            SELECT id, product_name, product_category, brand, price_per_unit, unit, 
                   package_size, tags, meal_components, embedding
            FROM products
            WHERE embedding IS NOT NULL
            AND price_per_unit < ?
        """
        
        # Фильтр по meal_component если есть
        if meal_components:
            main_component = meal_components[0] if isinstance(meal_components, list) else meal_components
            query += f" AND meal_components LIKE '%{main_component}%'"
        
        cursor.execute(query, (max_price,))
        rows = cursor.fetchall()
        
        if not rows:
            if close_conn:
                conn.close()
            return None
        
        # Считаем similarity для каждого кандидата
        candidates = []
        
        for row in rows:
            embedding_blob = row[9]
            if not embedding_blob:
                continue
            
            try:
                # Десериализуем embedding
                product_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                
                # Проверяем валидность
                if len(product_embedding) == 0:
                    continue
                
                if not np.isfinite(product_embedding).all():
                    continue
                
                # Проверяем исходный embedding
                if not np.isfinite(original_embedding).all():
                    continue
                
                # Semantic similarity
                similarity = float(cosine_similarity(
                    original_embedding.reshape(1, -1),
                    product_embedding.reshape(1, -1)
                )[0, 0])
                
                # Проверяем что similarity валидный
                if not np.isfinite(similarity):
                    continue
                
                candidates.append({
                    'id': row[0],
                    'name': row[1],
                    'product_name': row[1],
                    'product_category': row[2],
                    'brand': row[3],
                    'price': row[4],
                    'unit': row[5],
                    'package_size': row[6],
                    'tags': row[7],
                    'meal_components': row[8],
                    'embedding': product_embedding,
                    'similarity': similarity
                })
                
            except Exception as e:
                continue
        
        if close_conn:
            conn.close()
        
        if not candidates:
            return None
        
        # Сортируем по similarity (самые похожие вверху)
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Берём самый похожий (но не идентичный)
        for candidate in candidates:
            if candidate['id'] != item.get('id'):
                return candidate
        
        return None


# ==================== ТЕСТИРОВАНИЕ ====================

def test_budget_agent():
    """Тестирует работу BudgetAgent."""
    
    print("=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ BudgetAgent")
    print("=" * 70)
    
    agent = BudgetAgent()
    
    # Загружаем РЕАЛЬНЫЕ товары из БД
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, product_name, price_per_unit, embedding, meal_components
        FROM products
        WHERE embedding IS NOT NULL
        AND price_per_unit > 100
        ORDER BY price_per_unit DESC
        LIMIT 5
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("❌ Нет товаров с embeddings в БД!")
        return
    
    # Создаём корзину из реальных товаров
    expensive_basket = []
    for row in rows[:2]:
        embedding = np.frombuffer(row[3], dtype=np.float32)
        
        expensive_basket.append({
            'id': row[0],
            'name': row[1],
            'product_name': row[1],
            'price': row[2],
            'meal_components': row[4].split('|') if row[4] else ['main_course'],
            'embedding': embedding
        })
    
    print(f"\n📝 Тест 1: Дорогая корзина (бюджет 200₽)")
    for item in expensive_basket:
        print(f"   - {item['name'][:50]}: {item['price']:.2f}₽")
    
    result = agent.optimize(
        basket=expensive_basket,
        budget_rub=200.0,
        min_discount=0.2
    )
    
    print(f"\n📊 Результат:")
    print(f"   Исходная цена: {sum(i['price'] for i in expensive_basket):.2f}₽")
    print(f"   Итоговая цена: {result['total_price']:.2f}₽")
    print(f"   Сэкономлено: {result['saved']:.2f}₽")
    print(f"   В бюджете: {'✅' if result['within_budget'] else '❌'}")
    print(f"   Замен: {len(result['replacements'])}")
    
    for rep in result['replacements']:
        print(f"      {rep['from'][:40]} → {rep['to'][:40]} (-{rep['saved']:.2f}₽)")
    
    # Тест 2: Корзина в бюджете
    print("\n\n📝 Тест 2: Корзина в бюджете (бюджет 5000₽)")
    
    result2 = agent.optimize(
        basket=expensive_basket,
        budget_rub=5000.0
    )
    
    print(f"\n📊 Результат:")
    print(f"   {result2['message']}")
    print(f"   Замен: {len(result2['replacements'])}")
    
    print("\n" + "=" * 70)
    print("✅ Тестирование завершено")
    print("=" * 70)


if __name__ == "__main__":
    test_budget_agent()