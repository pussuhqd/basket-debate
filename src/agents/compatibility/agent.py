# src/agents/compatibility/agent.py
"""
Агент для подбора совместимых товаров на основе сценариев.
"""


from typing import Dict, List, Optional
from pathlib import Path


from src.agents.compatibility.scenario_matcher import ScenarioMatcher
from src.agents.compatibility.product_searcher import ProductSearcher
from src.agents.compatibility.scorer import CompatibilityScorer
from src.schemas.basket_item import BasketItem, create_basket_item



# ==================== КОНФИГУРАЦИЯ ====================


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SCENARIOS_PATH = PROJECT_ROOT / "data" / "scenarios.json"



# ==================== КЛАСС CompatibilityAgent ====================


class CompatibilityAgent:
    """
    Агент для генерации корзины на основе совместимости продуктов.
    
    Алгоритм:
    1. Выбирает сценарий (ScenarioMatcher)
    2. Ищет товары для каждого ингредиента (ProductSearcher)
    3. Оценивает совместимость корзины (CompatibilityScorer)
    """
    
    def __init__(self, scenarios_path: Path = SCENARIOS_PATH):
        """
        Инициализация агента.
        
        Args:
            scenarios_path: Путь к scenarios.json
        """
        print("=" * 70)
        print("🤖 ИНИЦИАЛИЗАЦИЯ CompatibilityAgent")
        print("=" * 70)
        
        # Загружаем компоненты
        self.scenario_matcher = ScenarioMatcher(scenarios_path=scenarios_path)
        self.searcher = ProductSearcher()  # ✅ БЕЗ db_path
        self.scorer = CompatibilityScorer()
        
        print("✅ CompatibilityAgent готов")
        print("=" * 70)
    
    
    def generate_basket(
        self,
        parsed_query: Dict,
        strategy: str = "smart"  # ← Изменили default
    ) -> Dict:
        """
        Генерирует корзину товаров с учётом предпочтений пользователя.
        """
        
        meal_types = parsed_query.get('meal_types', ['dinner'])
        people = parsed_query.get('people', 1)
        budget_rub = parsed_query.get('budget_rub')
        exclude_tags = parsed_query.get('exclude_tags', [])
        include_tags = parsed_query.get('include_tags', [])
        
        # ============================================
        # ШАГ 1: УМНЫЙ выбор сценария
        # ============================================
        
        max_time_min = parsed_query.get('max_time_min')
        prefer_quick = parsed_query.get('prefer_quick', False)
        prefer_cheap = parsed_query.get('prefer_cheap', False)
        if prefer_cheap == False:
            prefer_cheap = budget_rub is not None and budget_rub < 1000  # Если бюджет < 1000₽ - ищем дешёвое
        
        scenario = self.scenario_matcher.match(
            meal_types=meal_types,
            people=people,
            max_time_min=max_time_min,
            exclude_tags=exclude_tags,
            include_tags=include_tags,
            prefer_quick=prefer_quick,
            prefer_cheap=prefer_cheap,
            strategy="smart"
        )
    
        if not scenario:
            return {
                'success': False,
                'message': f'Не найдено сценариев для {meal_types} с тегами exclude={exclude_tags}, include={include_tags}',
                'basket': [],
                'total_price': 0
            }
        
        print(f"\n✅ Выбран сценарий: {scenario['name']}")
        print(f"   Учтены exclude_tags: {exclude_tags}")
        print(f"   Учтены include_tags: {include_tags}")

        # ============================================
        # ШАГ 2: Ищем товары для каждого ингредиента
        # ============================================
        basket = []
        total_price = 0.0
        
        for component in scenario['components']:
            ingredient = component['ingredient']
            search_query = component['search_query']
            quantity_needed = component.get('quantity_scaled', component['quantity_per_person'])
            unit = component['unit']
            required = component.get('required', True)
            
            print(f"\n🔍 Поиск: {ingredient} ({search_query})")
            
            # Поиск товаров
            candidates = self.searcher.search(
                query=search_query,
                limit=5,
                exclude_tags=exclude_tags,
                include_tags=include_tags
            )
            
            if not candidates and required:
                print(f"   ⚠️  Обязательный ингредиент не найден: {ingredient}")
                continue
            
            if not candidates:
                print(f"   ℹ️  Опциональный ингредиент пропущен: {ingredient}")
                continue
            
            # Берём лучший товар
            best_product = candidates[0]
            
            product_for_schema = {
                'id': best_product['id'],
                'name': best_product.get('product_name', best_product.get('name', '')),
                'price': best_product.get('price_per_unit', 0),
                'unit': best_product.get('unit', 'кг'),  # ✅ Уже нормализован
                'category': best_product.get('product_category', ''),
                'brand': best_product.get('brand', ''),
                'rating': best_product.get('rating')
            }

            # Конвертируем количество из сценария в единицы товара
            quantity_in_product_units = quantity_needed
            if unit == 'г' and product_for_schema['unit'] == 'кг':
                quantity_in_product_units = quantity_needed / 1000
            elif unit == 'мл' and product_for_schema['unit'] == 'л':
                quantity_in_product_units = quantity_needed / 1000
            # Если unit уже совпадает ('кг' == 'кг'), конвертация не нужна

            # Создаем BasketItem
            basket_item = create_basket_item(
                product=product_for_schema,
                quantity=quantity_in_product_units,  # уже в кг/л/шт
                agent='compatibility',
                reason=f'Найден по запросу "{search_query}"',
                ingredient_role=ingredient,
                search_score=best_product.get('score', 0)
            )

            
            basket.append(basket_item)
            total_price += basket_item['total_price']
            
            print(f"   ✅ {basket_item['name']}")
            print(f"      💰 Цена: {basket_item['price_per_unit']:.2f}₽/{basket_item['unit']}")
            print(f"      📦 Нужно: {basket_item['quantity']:.2f}{basket_item['unit']}")
            print(f"      💵 Итого: {basket_item['total_price']:.2f}₽")
        
        # ============================================
        # ШАГ 3: Оценка совместимости
        # ============================================
        compatibility_result = self.scorer.compute_score(basket)
        compatibility_score = compatibility_result['total_score']
        
        print(f"\n📊 Совместимость корзины: {compatibility_score:.2f}")
        print(f"💰 Итоговая цена: {total_price:.2f}₽")
        
        # Проверка бюджета
        within_budget = True
        if budget_rub and total_price > budget_rub:
            within_budget = False
            print(f"⚠️  Превышен бюджет: {total_price:.2f}₽ > {budget_rub}₽")
        
        return {
            'success': True,
            'basket': basket,
            'total_price': round(total_price, 2),
            'scenario_used': {
                'id': scenario.get('id'),
                'name': scenario.get('name'),
                'meal_type': scenario.get('meal_type'),
                'people': scenario.get('scaled_for_people')
            },
            'compatibility_score': round(compatibility_score, 4),
            'within_budget': within_budget,
            'compatibility_details': compatibility_result
        }



# ==================== ТЕСТИРОВАНИЕ ====================


def test_agent():
    """Тестирует работу CompatibilityAgent с умным выбором сценариев и тегами."""
    print("\n" + "=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ CompatibilityAgent")
    print("=" * 70)
    
    agent = CompatibilityAgent()
    
    # ---------------- Тест 1: базовый ужин ----------------
    print("\n📝 Тест 1: Ужин на двоих за 1500₽ (без ограничений)")
    
    query1 = {
        'meal_types': ['dinner'],
        'people': 2,
        'budget_rub': 1500,
        'exclude_tags': [],
        'include_tags': []
    }
    
    result1 = agent.generate_basket(query1, strategy="smart")
    
    print(f"\n{'='*70}")
    print("РЕЗУЛЬТАТ ТЕСТА 1:")
    print(f"{'='*70}")
    print(f"Успех: {result1['success']}")
    print(f"Сценарий: {result1['scenario_used']['name']}")
    print(f"Товаров: {len(result1['basket'])}")
    print(f"Итого: {result1['total_price']}₽")
    print(f"Совместимость: {result1['compatibility_score']}")
    print(f"В рамках бюджета: {result1['within_budget']}")
    
    print(f"\n📋 Корзина (первые 5 товаров):")
    for item in result1['basket'][:5]:
        print(f"   • {item['name']}")
        print(f"     {item['quantity']:.2f}{item['unit']} × {item['price_per_unit']:.2f}₽/{item['unit']} = {item['total_price']:.2f}₽")
    if len(result1['basket']) > 5:
        print(f"   ... и ещё {len(result1['basket']) - 5} товаров")
    
        print(f"\n{'='*70}")
    print("🧾 ДЕТАЛИЗИРОВАННЫЙ ЧЕК")
    print(f"{'='*70}")
    
    for i, item in enumerate(result1['basket'], 1):
        print(f"\n{i}. {item['name']}")
        print(f"   Роль: {item.get('ingredient_role', 'N/A')}")
        print(f"   ──────────────────────────────────────")
        print(f"   Цена за единицу:  {item['price_per_unit']:>8.2f} ₽/{item['unit']}")
        print(f"   Количество:       {item['quantity']:>8.2f} {item['unit']}")
        print(f"   ──────────────────────────────────────")
        print(f"   ИТОГО:            {item['total_price']:>8.2f} ₽")
    
    print(f"\n{'='*70}")
    print(f"ВСЕГО К ОПЛАТЕ:      {result1['total_price']:>8.2f} ₽")
    print(f"Количество позиций:  {len(result1['basket'])}")
    print(f"{'='*70}")

    # ---------------- Тест 2: ужин без молочки ----------------
    print("\n📝 Тест 2: Ужин без молочных продуктов (exclude_tags=['dairy'])")
    
    query2 = {
        'meal_types': ['dinner'],
        'people': 2,
        'budget_rub': 1500,
        'exclude_tags': ['dairy'],
        'include_tags': []
    }
    
    result2 = agent.generate_basket(query2, strategy="smart")
    
    print(f"\n{'='*70}")
    print("РЕЗУЛЬТАТ ТЕСТА 2:")
    print(f"{'='*70}")
    print(f"Успех: {result2['success']}")
    if result2['success']:
        print(f"Сценарий: {result2['scenario_used']['name']}")
        print(f"Товаров: {len(result2['basket'])}")
        print(f"Итого: {result2['total_price']}₽")
        
        dairy_keywords = ['молоко', 'сыр', 'творог', 'сметана',
                          'кефир', 'йогурт', 'ряженка', 'сливки']
        has_dairy = False
        for item in result2['basket']:
            name_lower = item['name'].lower()
            if any(k in name_lower for k in dairy_keywords):
                print(f"   ⚠ Найден молочный продукт: {item['name']}")
                has_dairy = True
        
        if not has_dairy:
            print("   ✅ Молочных продуктов нет (exclude_tags отработали корректно)")
    
    # ---------------- Тест 3: веганский ужин ----------------
    print("\n📝 Тест 3: Веганский ужин (без мяса, рыбы, молочки, include_tags=['vegan'])")
    
    query3 = {
        'meal_types': ['dinner'],
        'people': 2,
        'budget_rub': 1200,
        #'exclude_tags': ['meat','dairy'],
        'include_tags': ['vegan']
    }
    
    result3 = agent.generate_basket(query3, strategy="smart")
    
    print(f"\n{'='*70}")
    print("РЕЗУЛЬТАТ ТЕСТА 3:")
    print(f"{'='*70}")
    print(f"Успех: {result3['success']}")
    if result3['success']:
        print(f"Сценарий: {result3['scenario_used']['name']}")
        print(f"Товаров: {len(result3['basket'])}")
        print(f"Итого: {result3['total_price']}₽")
    
    print("\n" + "=" * 70)
    print("✅ Тестирование завершено")
    print("=" * 70)


if __name__ == "__main__":
    test_agent()
