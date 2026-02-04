"""
Модуль для подсчета compatibility score корзины товаров.

Оценивает насколько хорошо товары сочетаются друг с другом на основе:
- Семантического сходства (cosine similarity embeddings)
- Правил совместимости из meal_components_extended.json
- Позитивных и негативных пар продуктов

Использование:
    scorer = CompatibilityScorer()
    basket = [product1, product2, product3]
    score = scorer.compute_score(basket)
    # score: 0.0 (плохо) - 1.0 (отлично)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity


# ==================== КОНФИГУРАЦИЯ ====================

MEAL_COMPONENTS_PATH = Path("data/meal_components_extended.json")


# ==================== КЛАСС CompatibilityScorer ====================

class CompatibilityScorer:
    """
    Класс для оценки совместимости товаров в корзине.
    """
    
    def __init__(self, meal_components_path: Path = MEAL_COMPONENTS_PATH):
        """
        Инициализация scorer'а.
        
        Args:
            meal_components_path: Путь к meal_components_extended.json
        """
        self.meal_components_path = meal_components_path
        self.positive_pairs = []
        self.negative_pairs = []
        self.neutral_keywords = []
        self._load_compatibility_rules()
    
    
    def _load_compatibility_rules(self):
        """
        Загружает правила совместимости из JSON.
        """
        if not self.meal_components_path.exists():
            print(f"⚠️  Файл {self.meal_components_path} не найден")
            print("   Scorer будет работать только на основе embeddings")
            return
        
        with open(self.meal_components_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Извлекаем compatibility_matrix
        compatibility_matrix = data.get('compatibility_matrix', {})
        
        self.positive_pairs = compatibility_matrix.get('positive_pairs', [])
        self.negative_pairs = compatibility_matrix.get('negative_pairs', [])
        
        # Извлекаем neutral_pairs (товары, которые сочетаются со всем)
        neutral_pairs = compatibility_matrix.get('neutral_pairs', [])
        self.neutral_keywords = []
        for pair in neutral_pairs:
            if isinstance(pair, list) and len(pair) == 2:
                keyword, wildcard = pair
                if wildcard == "*":
                    self.neutral_keywords.append(keyword.lower())
        
        print(f"📊 Загружено правил совместимости:")
        print(f"   Позитивных пар: {len(self.positive_pairs)}")
        print(f"   Негативных пар: {len(self.negative_pairs)}")
        print(f"   Нейтральных ключевых слов: {len(self.neutral_keywords)}")
    
    
    def _check_pair_compatibility(
        self,
        product1_name: str,
        product2_name: str
    ) -> float:
        """
        Проверяет совместимость двух товаров по правилам.
        
        Args:
            product1_name: Название первого товара
            product2_name: Название второго товара
        
        Returns:
            float: Модификатор score
                +0.1 если позитивная пара
                -0.2 если негативная пара
                0.0 если нейтральная или нет правила
        """
        name1 = product1_name.lower()
        name2 = product2_name.lower()
        
        # Проверяем позитивные пары
        for pair in self.positive_pairs:
            if len(pair) != 2:
                continue
            
            keyword1, keyword2 = [k.lower() for k in pair]
            
            # Проверяем оба направления
            if (keyword1 in name1 and keyword2 in name2) or \
               (keyword1 in name2 and keyword2 in name1):
                return 0.1  # Бонус за хорошую пару
        
        # Проверяем негативные пары
        for pair in self.negative_pairs:
            if len(pair) != 2:
                continue
            
            keyword1, keyword2 = [k.lower() for k in pair]
            
            # Проверяем оба направления
            if (keyword1 in name1 and keyword2 in name2) or \
               (keyword1 in name2 and keyword2 in name1):
                return -0.2  # Штраф за плохую пару
        
        # Нейтральная пара
        return 0.0
    
    
    def _compute_embedding_similarity(self, basket: List[Dict]) -> float:
        """
        Вычисляет среднее косинусное сходство между embeddings товаров.
        
        Args:
            basket: Список товаров с полем 'embedding'
        
        Returns:
            float: Средний similarity score (0.0-1.0)
        """
        # Фильтруем товары с embeddings
        products_with_embeddings = [
            p for p in basket
            if 'embedding' in p and p['embedding'] is not None
        ]
        
        if len(products_with_embeddings) < 2:
            # Если меньше 2 товаров - нельзя считать similarity
            return 0.5  # Нейтральный score
        
        # Извлекаем embeddings
        embeddings = np.array([p['embedding'] for p in products_with_embeddings])
        
        # Нормализуем для cosine similarity
        embeddings_normalized = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        
        # Считаем матрицу similarity
        similarity_matrix = cosine_similarity(embeddings_normalized)
        
        # Берем только верхний треугольник (без диагонали)
        # Это все уникальные пары
        n = len(similarity_matrix)
        upper_triangle_indices = np.triu_indices(n, k=1)
        similarities = similarity_matrix[upper_triangle_indices]
        
        if len(similarities) == 0:
            return 0.5
        
        # Среднее сходство
        avg_similarity = float(np.mean(similarities))
        
        return avg_similarity
    
    
    def _compute_meal_component_balance(self, basket: List[Dict]) -> float:
        """
        Оценивает баланс meal_components в корзине.
        
        Хорошая корзина должна содержать:
        - main_course (основное блюдо)
        - side_dish (гарнир) или salad
        - Опционально: beverage, sauce
        
        Args:
            basket: Список товаров с полем 'meal_components'
        
        Returns:
            float: Balance score (0.0-1.0)
        """
        # Собираем все meal_components
        all_components = set()
        for product in basket:
            components = product.get('meal_components', [])
            if isinstance(components, list):
                all_components.update(components)
        
        score = 0.0
        
        # Проверяем наличие основных компонентов
        if 'main_course' in all_components:
            score += 0.4  # Основное блюдо очень важно
        
        if 'side_dish' in all_components or 'salad' in all_components:
            score += 0.3  # Гарнир или салат важны
        
        if 'beverage' in all_components:
            score += 0.1  # Напиток - бонус
        
        if 'sauce' in all_components:
            score += 0.1  # Соус - бонус
        
        if 'bakery' in all_components:
            score += 0.1  # Хлеб - бонус
        
        # Если слишком много компонентов одного типа - это плохо
        component_counts = {}
        for product in basket:
            for comp in product.get('meal_components', []):
                component_counts[comp] = component_counts.get(comp, 0) + 1
        
        # Штраф за дублирование main_course (два основных блюда - странно)
        if component_counts.get('main_course', 0) > 2:
            score -= 0.2
        
        # Нормализуем в диапазон [0, 1]
        return min(max(score, 0.0), 1.0)
    
    
    def compute_score(
        self,
        basket: List[Dict],
        weights: Dict[str, float] = None
    ) -> Dict:
        """
        Вычисляет итоговый compatibility score для корзины.
        
        Args:
            basket: Список товаров (каждый должен содержать:
                    'product_name', 'embedding', 'meal_components')
            weights: Веса для разных компонентов score:
                     - 'embedding_similarity': вес косинусного сходства
                     - 'rule_based': вес правил совместимости
                     - 'component_balance': вес баланса компонентов
        
        Returns:
            Dict: {
                'total_score': float (0.0-1.0),
                'embedding_similarity': float,
                'rule_based_modifier': float,
                'component_balance': float,
                'num_products': int,
                'num_positive_pairs': int,
                'num_negative_pairs': int
            }
        """
        # Веса по умолчанию
        if weights is None:
            weights = {
                'embedding_similarity': 0.5,
                'rule_based': 0.3,
                'component_balance': 0.2
            }
        
        if len(basket) == 0:
            return {
                'total_score': 0.0,
                'embedding_similarity': 0.0,
                'rule_based_modifier': 0.0,
                'component_balance': 0.0,
                'num_products': 0,
                'num_positive_pairs': 0,
                'num_negative_pairs': 0
            }
        
        # 1. Embedding similarity
        embedding_score = self._compute_embedding_similarity(basket)
        
        # 2. Rule-based compatibility
        rule_modifier = 0.0
        num_positive = 0
        num_negative = 0
        
        # Проверяем все пары товаров
        for i in range(len(basket)):
            for j in range(i + 1, len(basket)):
                product1 = basket[i]
                product2 = basket[j]
                
                pair_modifier = self._check_pair_compatibility(
                    product1.get('product_name', ''),
                    product2.get('product_name', '')
                )
                
                rule_modifier += pair_modifier
                
                if pair_modifier > 0:
                    num_positive += 1
                elif pair_modifier < 0:
                    num_negative += 1
        
        # Нормализуем rule_modifier (делим на количество пар)
        num_pairs = len(basket) * (len(basket) - 1) / 2
        if num_pairs > 0:
            rule_modifier_normalized = rule_modifier / num_pairs
        else:
            rule_modifier_normalized = 0.0
        
        # Конвертируем в диапазон [0, 1]
        # rule_modifier может быть от -0.2 до +0.1 на пару
        # Переводим: -0.2 → 0.0, 0.0 → 0.5, +0.1 → 1.0
        rule_score = max(0.0, min(1.0, 0.5 + rule_modifier_normalized * 2.5))
        
        # 3. Component balance
        balance_score = self._compute_meal_component_balance(basket)
        
        # 4. Итоговый score (взвешенная сумма)
        total_score = (
            embedding_score * weights['embedding_similarity'] +
            rule_score * weights['rule_based'] +
            balance_score * weights['component_balance']
        )
        
        # Нормализуем в [0, 1]
        total_score = max(0.0, min(1.0, total_score))
        
        return {
            'total_score': round(total_score, 4),
            'embedding_similarity': round(embedding_score, 4),
            'rule_based_score': round(rule_score, 4),
            'component_balance': round(balance_score, 4),
            'num_products': len(basket),
            'num_positive_pairs': num_positive,
            'num_negative_pairs': num_negative,
            'weights_used': weights
        }
    
    
    def get_score_interpretation(self, score: float) -> str:
        """
        Интерпретирует числовой score в текстовое описание.
        
        Args:
            score: Score от 0.0 до 1.0
        
        Returns:
            str: Текстовое описание
        """
        if score >= 0.8:
            return "Отличная совместимость 🌟"
        elif score >= 0.6:
            return "Хорошая совместимость ✅"
        elif score >= 0.4:
            return "Приемлемая совместимость 👍"
        elif score >= 0.2:
            return "Слабая совместимость ⚠️"
        else:
            return "Плохая совместимость ❌"


# ==================== ТЕСТИРОВАНИЕ ====================

def test_scorer():
    """
    Тестирует работу CompatibilityScorer.
    """
    print("=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ CompatibilityScorer")
    print("=" * 70)
    
    scorer = CompatibilityScorer()
    
    # Создаем тестовые товары с mock embeddings
    
    # Тест 1: Хорошая корзина (курица + рис + овощи)
    print("\n📝 Тест 1: Хорошая корзина (курица + рис + овощи)")
    
    basket1 = [
        {
            'product_name': 'Курица филе',
            'meal_components': ['main_course'],
            'embedding': np.random.randn(384) * 0.1 + np.array([1.0] * 384)
        },
        {
            'product_name': 'Рис круглозерный',
            'meal_components': ['side_dish'],
            'embedding': np.random.randn(384) * 0.1 + np.array([0.9] * 384)
        },
        {
            'product_name': 'Морковь',
            'meal_components': ['salad'],
            'embedding': np.random.randn(384) * 0.1 + np.array([0.95] * 384)
        },
        {
            'product_name': 'Масло растительное',
            'meal_components': ['sauce'],
            'embedding': np.random.randn(384) * 0.1 + np.array([0.85] * 384)
        }
    ]
    
    result1 = scorer.compute_score(basket1)
    
    print(f"\n   Total Score: {result1['total_score']} {scorer.get_score_interpretation(result1['total_score'])}")
    print(f"   - Embedding Similarity: {result1['embedding_similarity']}")
    print(f"   - Rule-based Score: {result1['rule_based_score']}")
    print(f"   - Component Balance: {result1['component_balance']}")
    print(f"   - Positive Pairs: {result1['num_positive_pairs']}")
    print(f"   - Negative Pairs: {result1['num_negative_pairs']}")
    
    # Тест 2: Плохая корзина (молоко + рыба)
    print("\n\n📝 Тест 2: Плохая корзина (молоко + рыба)")
    
    basket2 = [
        {
            'product_name': 'Молоко 3.2%',
            'meal_components': ['beverage'],
            'embedding': np.random.randn(384) * 0.3 + np.array([1.0] * 384)
        },
        {
            'product_name': 'Рыба филе',
            'meal_components': ['main_course'],
            'embedding': np.random.randn(384) * 0.3 + np.array([-0.5] * 384)
        }
    ]
    
    result2 = scorer.compute_score(basket2)
    
    print(f"\n   Total Score: {result2['total_score']} {scorer.get_score_interpretation(result2['total_score'])}")
    print(f"   - Embedding Similarity: {result2['embedding_similarity']}")
    print(f"   - Rule-based Score: {result2['rule_based_score']}")
    print(f"   - Component Balance: {result2['component_balance']}")
    print(f"   - Negative Pairs: {result2['num_negative_pairs']} (молоко + рыба)")
    
    # Тест 3: Несбалансированная корзина (только снеки)
    print("\n\n📝 Тест 3: Несбалансированная корзина (только снеки)")
    
    basket3 = [
        {
            'product_name': 'Чипсы',
            'meal_components': ['snack'],
            'embedding': np.random.randn(384) * 0.2
        },
        {
            'product_name': 'Печенье',
            'meal_components': ['snack'],
            'embedding': np.random.randn(384) * 0.2
        },
        {
            'product_name': 'Орехи',
            'meal_components': ['snack'],
            'embedding': np.random.randn(384) * 0.2
        }
    ]
    
    result3 = scorer.compute_score(basket3)
    
    print(f"\n   Total Score: {result3['total_score']} {scorer.get_score_interpretation(result3['total_score'])}")
    print(f"   - Component Balance: {result3['component_balance']} (нет main_course)")
    
    # Тест 4: Пустая корзина
    print("\n\n📝 Тест 4: Пустая корзина")
    
    result4 = scorer.compute_score([])
    print(f"\n   Total Score: {result4['total_score']}")
    
    # Тест 5: Кастомные веса
    print("\n\n📝 Тест 5: Кастомные веса (приоритет на правила)")
    
    custom_weights = {
        'embedding_similarity': 0.2,
        'rule_based': 0.6,
        'component_balance': 0.2
    }
    
    result5 = scorer.compute_score(basket1, weights=custom_weights)
    
    print(f"\n   Total Score: {result5['total_score']}")
    print(f"   Weights: {result5['weights_used']}")
    
    print("\n" + "=" * 70)
    print("✅ Тестирование завершено")
    print("=" * 70)


if __name__ == "__main__":
    test_scorer()
