"""
Главный модуль агента Compatibility.

Агент отвечает за создание базовой продуктовой корзины на основе:
- Запроса пользователя (meal_types, people, budget)
- Сценариев блюд (scenarios.json)
- Семантического поиска товаров (embeddings)
- Оценки совместимости товаров (compatibility score)

Использование:
    agent = CompatibilityAgent()
    result = agent.generate_basket({
        "meal_types": ["dinner"],
        "people": 3,
        "budget_rub": 2000,
        "exclude_tags": [],
        "include_tags": []
    })
"""

from pathlib import Path
from typing import Dict, List, Optional
import time

from src.agent.compatibility.scenario_matcher import ScenarioMatcher
from src.agent.compatibility.product_searcher import ProductSearcher
from src.agent.compatibility.scorer import CompatibilityScorer


# ==================== КОНФИГУРАЦИЯ ====================

# Пути по умолчанию
DB_PATH = Path("data/processed/products.db")
SCENARIOS_PATH = Path("data/scenarios.json")
MEAL_COMPONENTS_PATH = Path("data/meal_components_extended.json")


# ==================== КЛАСС CompatibilityAgent ====================

class CompatibilityAgent:
    """
    Агент для генерации базовой продуктовой корзины.
    
    Работает в 4 этапа:
    1. Выбор сценария блюда (ScenarioMatcher)
    2. Поиск товаров для каждого ингредиента (ProductSearcher)
    3. Формирование корзины
    4. Оценка совместимости (CompatibilityScorer)
    """
    
    def __init__(
        self,
        db_path: Path = DB_PATH,
        scenarios_path: Path = SCENARIOS_PATH,
        meal_components_path: Path = MEAL_COMPONENTS_PATH
    ):
        """
        Инициализация агента.
        
        Args:
            db_path: Путь к БД с товарами
            scenarios_path: Путь к scenarios.json
            meal_components_path: Путь к meal_components_extended.json
        """
        print("=" * 70)
        print("🤖 ИНИЦИАЛИЗАЦИЯ CompatibilityAgent")
        print("=" * 70)
        
        # Инициализируем компоненты
        self.matcher = ScenarioMatcher(scenarios_path=scenarios_path)
        self.searcher = ProductSearcher(db_path=db_path)
        self.scorer = CompatibilityScorer(meal_components_path=meal_components_path)
        
        print("=" * 70)
        print("✅ Агент готов к работе")
        print("=" * 70)
    
    
    def generate_basket(
        self,
        parsed_query: Dict,
        strategy: str = "random",
        max_time_min: Optional[int] = None
    ) -> Dict:
        """
        Генерирует продуктовую корзину на основе запроса.
        
        Args:
            parsed_query: Распарсенный запрос от NLP парсера:
                {
                    "meal_types": ["dinner"],
                    "people": 3,
                    "budget_rub": 2000,
                    "exclude_tags": ["dairy"],
                    "include_tags": []
                }
            strategy: Стратегия выбора сценария (random, fastest, simplest)
            max_time_min: Максимальное время приготовления
        
        Returns:
            Dict: {
                "success": bool,
                "basket": List[Dict],  # Список товаров
                "scenario_used": Dict,  # Использованный сценарий
                "compatibility_score": Dict,  # Оценка совместимости
                "total_price": float,
                "budget_rub": float,
                "people": int,
                "execution_time_sec": float,
                "errors": List[str]
            }
        """
        start_time = time.time()
        
        errors = []
        basket = []
        scenario_used = None
        compatibility_result = None
        
        try:
            # 1. Извлекаем параметры из запроса
            meal_types = parsed_query.get('meal_types', ['dinner'])
            people = parsed_query.get('people', 1)
            budget_rub = parsed_query.get('budget_rub')
            exclude_tags = parsed_query.get('exclude_tags', [])
            include_tags = parsed_query.get('include_tags', [])
            
            print(f"\n🔍 Генерация корзины...")
            print(f"   Meal types: {meal_types}")
            print(f"   People: {people}")
            print(f"   Budget: {budget_rub}₽" if budget_rub else "   Budget: не указан")
            print(f"   Exclude tags: {exclude_tags}" if exclude_tags else "")
            print(f"   Include tags: {include_tags}" if include_tags else "")
            
            # 2. Выбираем сценарий
            print(f"\n📋 Этап 1: Выбор сценария ({strategy})...")
            
            scenario = self.matcher.match(
                meal_types=meal_types if meal_types else None,
                people=people,
                max_time_min=max_time_min,
                strategy=strategy
            )
            
            if not scenario:
                error_msg = f"Не найдено сценариев для meal_types={meal_types}"
                errors.append(error_msg)
                print(f"   ❌ {error_msg}")
                
                return {
                    "success": False,
                    "basket": [],
                    "scenario_used": None,
                    "compatibility_score": None,
                    "total_price": 0.0,
                    "budget_rub": budget_rub,
                    "people": people,
                    "execution_time_sec": time.time() - start_time,
                    "errors": errors
                }
            
            scenario_used = scenario
            print(f"   ✅ Выбран: {scenario['name']}")
            print(f"      Ингредиентов: {len(scenario['components'])}")
            print(f"      Время приготовления: {scenario['estimated_time_min']} мин")
            
            # 3. Ищем товары для каждого ингредиента
            print(f"\n🔎 Этап 2: Поиск товаров...")
            
            for i, component in enumerate(scenario['components'], 1):
                ingredient = component['ingredient']
                search_query = component['search_query']
                meal_component = component['meal_component']
                required = component['required']
                quantity_scaled = component.get('quantity_scaled', component['quantity_per_person'])
                
                print(f"\n   {i}. Ищем '{ingredient}' (query: '{search_query}')")
                
                # Ищем товар
                product = self.searcher.search_by_ingredient(
                    ingredient_name=search_query,
                    quantity_grams=quantity_scaled,
                    meal_component=meal_component,
                    people=1  # Уже масштабировано в сценарии
                )
                
                if product:
                    print(f"      ✅ Найден: {product['product_name']}")
                    print(f"         Цена: {product['total_price']}₽ "
                          f"({product['quantity_needed']}x{product['package_size']}{product['unit']})")
                    print(f"         Score: {product['search_score']:.3f}")
                    
                    # Добавляем метаданные о компоненте
                    product['ingredient_role'] = ingredient
                    product['meal_component'] = meal_component
                    product['required'] = required
                    
                    basket.append(product)
                    if 'quantity_grams_per_person' in product and 'package_size' in product:
                        actual_needed_grams = product['quantity_grams_per_person'] * people
                        package_grams = product['package_size'] * (1000 if product['unit'] == 'кг' else 1)
                        
                        fractional_cost = (actual_needed_grams / package_grams) * product['price_per_unit']
                        product['fractional_cost'] = round(fractional_cost, 2)
                        product['actual_needed_grams'] = actual_needed_grams

                else:
                    warning = f"Товар не найден для '{ingredient}'"
                    errors.append(warning)
                    print(f"      ⚠️  {warning}")
                    
                    if required:
                        print(f"         (обязательный ингредиент!)")
            
            # 4. Проверяем бюджет
            total_price = sum(p['total_price'] for p in basket)
            print(f"\n💰 Этап 3: Проверка бюджета...")
            print(f"   Итого: {total_price:.2f}₽")
            
            if budget_rub:
                print(f"   Бюджет: {budget_rub}₽")
                if total_price > budget_rub:
                    print(f"   ⚠️  Превышен на {total_price - budget_rub:.2f}₽")
                    errors.append(f"Превышен бюджет: {total_price:.2f}₽ > {budget_rub}₽")
                else:
                    print(f"   ✅ В пределах бюджета (запас: {budget_rub - total_price:.2f}₽)")
            
            # 5. Оцениваем совместимость
            print(f"\n🎯 Этап 4: Оценка совместимости...")
            
            compatibility_result = self.scorer.compute_score(basket)
            
            print(f"   Total Score: {compatibility_result['total_score']:.3f} "
                  f"{self.scorer.get_score_interpretation(compatibility_result['total_score'])}")
            print(f"   - Embedding Similarity: {compatibility_result['embedding_similarity']:.3f}")
            print(f"   - Rule-based: {compatibility_result['rule_based_score']:.3f}")
            print(f"   - Component Balance: {compatibility_result['component_balance']:.3f}")
            
            if compatibility_result['num_negative_pairs'] > 0:
                print(f"   ⚠️  Негативных пар: {compatibility_result['num_negative_pairs']}")
                errors.append(f"Обнаружено {compatibility_result['num_negative_pairs']} несовместимых пар")
            
            # 6. Формируем результат
            execution_time = time.time() - start_time
            
            success = (
                len(basket) > 0 and
                (budget_rub is None or total_price <= budget_rub) and
                compatibility_result['total_score'] >= 0.3
            )
            
            print(f"\n{'='*70}")
            if success:
                print(f"✅ Корзина успешно сгенерирована за {execution_time:.2f}с")
            else:
                print(f"⚠️  Корзина сгенерирована с предупреждениями за {execution_time:.2f}с")
            print(f"{'='*70}")
            
            return {
                "success": success,
                "basket": basket,
                "scenario_used": {
                    "id": scenario_used['id'],
                    "name": scenario_used['name'],
                    "meal_type": scenario_used['meal_type'],
                    "estimated_time_min": scenario_used['estimated_time_min']
                },
                "compatibility_score": compatibility_result,
                "total_price": round(total_price, 2),
                "budget_rub": budget_rub,
                "people": people,
                "execution_time_sec": round(execution_time, 2),
                "errors": errors,
                "warnings": errors  # Для обратной совместимости
            }
        
        except Exception as e:
            print(f"\n❌ Ошибка при генерации корзины: {e}")
            errors.append(str(e))
            
            return {
                "success": False,
                "basket": basket,
                "scenario_used": scenario_used,
                "compatibility_score": compatibility_result,
                "total_price": sum(p['total_price'] for p in basket) if basket else 0.0,
                "budget_rub": parsed_query.get('budget_rub'),
                "people": parsed_query.get('people', 1),
                "execution_time_sec": time.time() - start_time,
                "errors": errors
            }
    
    
    def generate_basket_simple(
        self,
        meal_type: str = "dinner",
        people: int = 2,
        budget_rub: Optional[float] = None
    ) -> Dict:
        """
        Упрощенный метод генерации корзины (без полного запроса).
        
        Args:
            meal_type: Тип приема пищи
            people: Количество человек
            budget_rub: Бюджет в рублях
        
        Returns:
            Dict: Результат генерации
        """
        parsed_query = {
            "meal_types": [meal_type],
            "people": people,
            "budget_rub": budget_rub,
            "exclude_tags": [],
            "include_tags": []
        }
        
        return self.generate_basket(parsed_query)


# ==================== ТЕСТИРОВАНИЕ ====================

def test_agent():
    """
    Тестирует работу CompatibilityAgent end-to-end.
    """
    print("\n" + "=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ CompatibilityAgent (END-TO-END)")
    print("=" * 70)
    
    agent = CompatibilityAgent()
    
    # Тест 1: Простой запрос - ужин на 3 человек
    print("\n\n" + "=" * 70)
    print("📝 ТЕСТ 1: Ужин на 3 человек за 2000₽")
    print("=" * 70)
    
    result1 = agent.generate_basket({
        "meal_types": ["dinner"],
        "people": 3,
        "budget_rub": 2000,
        "exclude_tags": [],
        "include_tags": []
    })
    
    print("\n📊 Результат:")
    print(f"   Success: {result1['success']}")
    print(f"   Товаров в корзине: {len(result1['basket'])}")
    print(f"   Сценарий: {result1['scenario_used']['name']}")
    print(f"   Итоговая цена: {result1['total_price']}₽")
    print(f"   Compatibility Score: {result1['compatibility_score']['total_score']:.3f}")
    print(f"   Время выполнения: {result1['execution_time_sec']}с")
    
    if result1['errors']:
        print(f"   ⚠️  Предупреждения: {len(result1['errors'])}")
    
    # Тест 2: Быстрый завтрак
    print("\n\n" + "=" * 70)
    print("📝 ТЕСТ 2: Быстрый завтрак на 1 человека")
    print("=" * 70)
    
    result2 = agent.generate_basket_simple(
        meal_type="breakfast",
        people=1,
        budget_rub=500
    )
    
    print("\n📊 Результат:")
    print(f"   Сценарий: {result2['scenario_used']['name']}")
    print(f"   Товаров: {len(result2['basket'])}")
    print(f"   Цена: {result2['total_price']}₽")
    
    # Тест 3: Обед на большую компанию
    print("\n\n" + "=" * 70)
    print("📝 ТЕСТ 3: Обед на 6 человек")
    print("=" * 70)
    
    result3 = agent.generate_basket_simple(
        meal_type="lunch",
        people=6,
        budget_rub=3000
    )
    
    print("\n📊 Результат:")
    print(f"   Сценарий: {result3['scenario_used']['name']}")
    print(f"   Товаров: {len(result3['basket'])}")
    print(f"   Цена: {result3['total_price']}₽")
    print(f"   Component Balance: {result3['compatibility_score']['component_balance']:.3f}")
    
    print("\n" + "=" * 70)
    print("✅ Все тесты завершены")
    print("=" * 70)


if __name__ == "__main__":
    test_agent()
