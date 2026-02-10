"""
Модуль для УМНОГО выбора сценария блюда на основе запроса пользователя.

Новые возможности:
- Фильтрация по exclude_tags (без молока, без мяса)
- Приоритизация по include_tags (веганское, халяль)
- Scoring система (учитывает время, стоимость, соответствие запросу)
- Поддержка "быстро/дешево"

Использование:
    matcher = ScenarioMatcher()
    scenario = matcher.match(
        meal_types=["dinner"],
        people=3,
        exclude_tags=["dairy", "meat"],
        include_tags=["vegan"],
        prefer_quick=True
    )
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random
from copy import deepcopy
from random import randint

SCENARIOS_PATH = Path("data/scenarios.json")

# Маппинг тегов на ключевые слова в ингредиентах
TAG_KEYWORDS = {
    'dairy': ['молоко', 'сыр', 'творог', 'сметана', 'кефир', 'йогурт', 'ряженка', 'масло сливочное'],
    'meat': ['курица', 'говядина', 'свинина', 'баранина', 'мясо', 'фарш', 'колбаса', 'сосиски'],
    'fish': ['рыба', 'лосось', 'треска', 'тунец', 'морепродукты', 'креветки'],
    'gluten': ['мука', 'хлеб', 'макароны', 'паста', 'лапша', 'булка'],
    'no_sugar': ['сахар', 'мёд', 'шоколад', 'варенье'],
    'alcohol': ['вино', 'пиво', 'водка', 'коньяк'],
    
    # Позитивные теги (что ДОЛЖНО быть)
    'vegan': ['овощи', 'фрукты', 'крупа', 'бобовые', 'нут', 'чечевица', 'тофу'],
    'vegetarian': ['овощи', 'фрукты', 'яйца', 'молоко', 'сыр'],
    'halal': ['курица', 'говядина', 'баранина', 'овощи', 'крупа'],
    'children_goods': ['каша', 'молоко', 'фрукты', 'йогурт']
}

# Примерная стоимость категорий (для быстрой оценки "дешево/дорого")
INGREDIENT_COST_ESTIMATE = {
    'курица': 500,
    'говядина': 600,
    'рыба': 800,
    'овощи': 300,
    'крупа': 180,
    'молоко': 190,
    'сыр': 900,
    'фрукты': 500
}

# ==================== КЛАСС ScenarioMatcher ====================

class ScenarioMatcher:
    """
    Класс для УМНОГО выбора сценария блюда из библиотеки сценариев.
    """
    
    def __init__(self, scenarios_path: Path = SCENARIOS_PATH):
        """
        Инициализация matcher'а.
        
        Args:
            scenarios_path: Путь к файлу scenarios.json
        """
        self.scenarios_path = scenarios_path
        self.scenarios = []
        self._load_scenarios()
    
    def _load_scenarios(self):
        """Загружает сценарии из JSON файла."""
        if not self.scenarios_path.exists():
            raise FileNotFoundError(
                f"Файл сценариев не найден: {self.scenarios_path}\n"
                f"Убедитесь, что вы создали data/scenarios.json"
            )
        
        with open(self.scenarios_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.scenarios = data.get('scenarios', [])
        
        if not self.scenarios:
            raise ValueError("Файл scenarios.json не содержит сценариев!")
        
        print(f"📚 Загружено {len(self.scenarios)} сценариев")
        
        # Статистика
        meal_type_counts = {}
        for scenario in self.scenarios:
            meal_type = scenario.get('meal_type', 'unknown')
            meal_type_counts[meal_type] = meal_type_counts.get(meal_type, 0) + 1
        
        print(f"   Распределение по типам:")
        for meal_type, count in sorted(meal_type_counts.items()):
            print(f"     - {meal_type}: {count}")
    
    def _check_ingredient_has_tag(self, ingredient_name: str, tag: str) -> bool:
        """
        Проверяет, содержит ли ингредиент указанный тег.
        
        Args:
            ingredient_name: Название ингредиента (например, "молоко")
            tag: Тег для проверки (например, "dairy")
        
        Returns:
            bool: True если ингредиент содержит этот тег
        """
        ingredient_lower = ingredient_name.lower()
        
        keywords = TAG_KEYWORDS.get(tag, [])
        
        for keyword in keywords:
            if keyword in ingredient_lower:
                return True
        
        return False
    
    def _filter_by_tags(
        self,
        scenarios: List[Dict],
        exclude_tags: List[str],
        include_tags: List[str]
    ) -> List[Dict]:
        """
        Фильтрует сценарии по exclude_tags и include_tags.
        
        Args:
            scenarios: Список сценариев
            exclude_tags: Теги для исключения (например, ["dairy", "meat"])
            include_tags: Теги для включения (например, ["vegan"])
        
        Returns:
            List[Dict]: Отфильтрованные сценарии
        """
        filtered = []
        
        for scenario in scenarios:
            components = scenario.get('components', [])
            
            # 1. Проверка exclude_tags (если хотя бы один ингредиент содержит запрещённый тег - убираем сценарий)
            has_excluded = False
            for component in components:
                ingredient = component.get('ingredient', '')
                
                for exclude_tag in exclude_tags:
                    if self._check_ingredient_has_tag(ingredient, exclude_tag):
                        has_excluded = True
                        break
                
                if has_excluded:
                    break
            
            if has_excluded:
                continue  # Этот сценарий не подходит
            
            # 2. Проверка include_tags (если указаны - хотя бы один ингредиент должен содержать нужный тег)
            if include_tags:
                has_included = False
                for component in components:
                    ingredient = component.get('ingredient', '')
                    
                    for include_tag in include_tags:
                        if self._check_ingredient_has_tag(ingredient, include_tag):
                            has_included = True
                            break
                    
                    if has_included:
                        break
                
                if not has_included:
                    continue  # Этот сценарий не содержит нужных ингредиентов
            
            # Сценарий прошёл фильтрацию
            filtered.append(scenario)
        
        return filtered
    
    def _compute_scenario_score(
        self,
        scenario: Dict,
        prefer_quick: bool = False,
        prefer_cheap: bool = False,
        include_tags: List[str] = None
    ) -> float:
        """
        Вычисляет score сценария на основе предпочтений.
        
        Args:
            scenario: Сценарий
            prefer_quick: Приоритет на быстрое приготовление
            prefer_cheap: Приоритет на дешевизну
            include_tags: Теги для бонусов
        
        Returns:
            float: Score (чем выше, тем лучше)
        """
        score = 1.0  # Базовый score
        
        # 1. Бонус за быстроту (если prefer_quick=True)
        if prefer_quick:
            time_min = scenario.get('estimated_time_min', 60)
            
            if time_min <= 15:
                score += 0.5  # Очень быстро
            elif time_min <= 30:
                score += 0.3  # Быстро
            elif time_min <= 45:
                score += 0.1  # Средне
            else:
                score -= 0.2  # Долго
        
        # 2. Бонус за дешевизну (если prefer_cheap=True)
        if prefer_cheap:
            components = scenario.get('components', [])
            
            # Примерная оценка стоимости на основе ингредиентов
            estimated_cost = 0
            for component in components:
                ingredient_lower = component.get('ingredient', '').lower()
                
                # Ищем примерную стоимость
                for key, cost in INGREDIENT_COST_ESTIMATE.items():
                    if key in ingredient_lower:
                        estimated_cost += cost
                        break
                else:
                    # Если не нашли - предполагаем среднюю стоимость
                    estimated_cost += 150
            
            # Чем дешевле - тем лучше
            if estimated_cost < 500:
                score += 0.4
            elif estimated_cost < 800:
                score += 0.2
            elif estimated_cost > 1200:
                score -= 0.2
        
        # 3. Бонус за соответствие include_tags
        if include_tags:
            components = scenario.get('components', [])
            
            matches = 0
            for component in components:
                ingredient = component.get('ingredient', '')
                
                for include_tag in include_tags:
                    if self._check_ingredient_has_tag(ingredient, include_tag):
                        matches += 1
                        break
            
            # Чем больше совпадений - тем выше score
            score += 0.1 * matches
        
        # 4. Штраф за слишком много ингредиентов (сложность)
        num_components = len(scenario.get('components', []))
        if num_components > 10:
            score -= 0.2
        
        return score
    
    def match(
        self,
        meal_types: Optional[List[str]] = None,
        people: int = 1,
        max_time_min: Optional[int] = None,
        exclude_tags: Optional[List[str]] = None,
        include_tags: Optional[List[str]] = None,
        prefer_quick: bool = False,
        prefer_cheap: bool = False,
        strategy: str = "smart"
    ) -> Optional[Dict]:
        """
        УМНЫЙ выбор сценария на основе запроса пользователя.
        
        Args:
            meal_types: Типы приемов пищи (например, ["dinner"])
            people: Количество человек
            max_time_min: Максимальное время приготовления
            exclude_tags: Теги для исключения (["dairy", "meat"])
            include_tags: Теги для включения (["vegan"])
            prefer_quick: Приоритет на быстрое приготовление
            prefer_cheap: Приоритет на дешевизну
            strategy: Стратегия выбора:
                - "smart" (по умолчанию) - выбирает сценарий с максимальным score
                - "random" - случайный из подходящих
                - "fastest" - самый быстрый
                - "simplest" - с минимумом ингредиентов
        
        Returns:
            Dict: Выбранный сценарий с масштабированными количествами
                  или None если не найдено подходящих
        """
        # 1. Базовая фильтрация по meal_types и времени
        candidates = self._filter_scenarios(
            meal_types=meal_types,
            max_time_min=max_time_min
        )
        
        if not candidates:
            print(f"⚠️  Не найдено сценариев для meal_types={meal_types}, max_time={max_time_min}")
            return None
        
        print(f"   🔍 После базовой фильтрации: {len(candidates)} сценариев")
        
        # 2. Фильтрация по exclude_tags и include_tags
        if exclude_tags or include_tags:
            candidates = self._filter_by_tags(
                scenarios=candidates,
                exclude_tags=exclude_tags or [],
                include_tags=include_tags or []
            )
            
            print(f"   🏷️  После фильтрации по тегам: {len(candidates)} сценариев")
            
            if not candidates:
                print(f"   ⚠️  Не найдено сценариев с учётом exclude_tags={exclude_tags}, include_tags={include_tags}")
                return None
        
        # 3. Выбор сценария по стратегии
        if strategy == "smart":
            # Вычисляем score для каждого сценария
            scored_scenarios = []
            for scenario in candidates:
                score = self._compute_scenario_score(
                    scenario=scenario,
                    prefer_quick=prefer_quick,
                    prefer_cheap=prefer_cheap,
                    include_tags=include_tags or []
                )
                scored_scenarios.append((scenario, score))
            
            # Сортируем по убыванию score
            scored_scenarios.sort(key=lambda x: x[1], reverse=True)
            
            # Берём топ-1 randomm
            r_ind = randint(0,min(5,len(scored_scenarios)))
            selected, best_score = scored_scenarios[r_ind]
            
            print(f"   ⭐ Выбран сценарий с score={best_score:.2f}: {selected['name']}")
        
        elif strategy == "random":
            selected = random.choice(candidates)
        
        elif strategy == "fastest":
            selected = min(candidates, key=lambda s: s.get('estimated_time_min', 999))
        
        elif strategy == "simplest":
            selected = min(candidates, key=lambda s: len(s.get('components', [])))
        
        else:
            print(f"⚠️  Неизвестная стратегия '{strategy}', используется 'smart'")
            selected = random.choice(candidates)
        
        # 4. Масштабируем под количество людей
        scaled_scenario = self._scale_scenario(selected, people)
        
        return scaled_scenario
    
    def _filter_scenarios(
        self,
        meal_types: Optional[List[str]] = None,
        max_time_min: Optional[int] = None,
        min_serves: Optional[int] = None
    ) -> List[Dict]:
        """Базовая фильтрация сценариев (без изменений)."""
        filtered = self.scenarios.copy()
        
        if meal_types:
            filtered = [s for s in filtered if s.get('meal_type') in meal_types]
        
        if max_time_min is not None:
            filtered = [s for s in filtered if s.get('estimated_time_min', 999) <= max_time_min]
        
        if min_serves is not None:
            filtered = [s for s in filtered if s.get('serves_base', 1) >= min_serves]
        
        return filtered
    
    def _scale_scenario(self, scenario: Dict, people: int) -> Dict:
        """Масштабирует количество ингредиентов под количество людей."""
        scaled_scenario = deepcopy(scenario)
        
        for component in scaled_scenario.get('components', []):
            quantity_per_person = component['quantity_per_person']
            
            # Умножаем на количество людей
            scaled_quantity = quantity_per_person * people
            
            # Округляем для удобства
            if scaled_quantity < 10:
                scaled_quantity = round(scaled_quantity, 1)
            elif scaled_quantity < 100:
                scaled_quantity = round(scaled_quantity / 5) * 5
            else:
                scaled_quantity = round(scaled_quantity / 10) * 10
            
            component['quantity_scaled'] = max(scaled_quantity, 1)
        
        scaled_scenario['scaled_for_people'] = people
        scaled_scenario['original_serves_base'] = scenario.get('serves_base', 1)
        
        return scaled_scenario

    
    def get_scenario_by_id(self, scenario_id: str, people: int = 1) -> Optional[Dict]:
        """
        Получает сценарий по его ID.
        
        Args:
            scenario_id: ID сценария (например, "dinner_chicken_vegetables")
            people: Количество человек для масштабирования
        
        Returns:
            Dict: Сценарий или None если не найден
        """
        scenario = next(
            (s for s in self.scenarios if s.get('id') == scenario_id),
            None
        )
        
        if scenario:
            return self._scale_scenario(scenario, people)
        
        return None
    
    
    def get_all_scenarios(self, meal_type: Optional[str] = None) -> List[Dict]:
        """
        Возвращает все сценарии (опционально отфильтрованные по meal_type).
        
        Args:
            meal_type: Тип приема пищи для фильтрации
        
        Returns:
            List[Dict]: Список сценариев
        """
        if meal_type:
            return [s for s in self.scenarios if s.get('meal_type') == meal_type]
        return self.scenarios.copy()
    
    
    def get_scenario_summary(self, scenario: Dict) -> str:
        """
        Создает текстовое описание сценария.
        
        Args:
            scenario: Сценарий
        
        Returns:
            str: Текстовое описание
        """
        name = scenario.get('name', 'Без названия')
        meal_type = scenario.get('meal_type', 'unknown')
        time_min = scenario.get('estimated_time_min', '?')
        people = scenario.get('scaled_for_people', scenario.get('serves_base', 1))
        
        components = scenario.get('components', [])
        num_components = len(components)
        
        # Список основных ингредиентов (только required)
        main_ingredients = [
            c['ingredient'] for c in components
            if c.get('required', True)
        ]
        
        summary = f"""
        Сценарий: {name}
        Тип: {meal_type}
        Время приготовления: {time_min} мин
        Порций: {people}
        Ингредиентов: {num_components}
        Основные: {', '.join(main_ingredients[:5])}
        """.strip()
        
        return summary


# ==================== ТЕСТИРОВАНИЕ ====================

def test_scenario_matcher():
    """
    Тестирует работу ScenarioMatcher.
    """
    print("=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ ScenarioMatcher")
    print("=" * 70)
    
    matcher = ScenarioMatcher()
    
    # Тест 1: Выбор сценария для ужина
    print("\n📝 Тест 1: Выбор сценария для ужина на 3 человек")
    scenario = matcher.match(
        meal_types=["dinner"],
        people=3,
        strategy="random"
    )
    
    if scenario:
        print(f"\n✅ Выбран сценарий: {scenario['name']}")
        print(f"   ID: {scenario['id']}")
        print(f"   Описание: {scenario['description']}")
        print(f"   Базовых порций: {scenario['serves_base']}")
        print(f"   Масштабировано на: {scenario['scaled_for_people']} чел.")
        print(f"   Коэффициент масштабирования: {scenario['scale_factor']:.2f}")
        print(f"   Время приготовления: {scenario['estimated_time_min']} мин")
        
        print(f"\n   📋 Ингредиенты:")
        for comp in scenario['components']:
            original = comp['quantity_per_person']
            scaled = comp.get('quantity_scaled', original)
            required = "✓" if comp['required'] else "○"
            
            print(f"      {required} {comp['ingredient']}: "
                  f"{original}{comp['unit']}/чел → {scaled}{comp['unit']} (всего)")
            print(f"        Поиск: '{comp['search_query']}'")
    else:
        print("❌ Сценарий не найден")
    
    # Тест 2: Самый быстрый завтрак
    print("\n\n📝 Тест 2: Самый быстрый завтрак")
    scenario = matcher.match(
        meal_types=["breakfast"],
        people=1,
        strategy="fastest"
    )
    
    if scenario:
        print(f"\n✅ {scenario['name']}")
        print(f"   Время: {scenario['estimated_time_min']} мин")
        print(matcher.get_scenario_summary(scenario))
    
    # Тест 3: Обед с ограничением по времени
    print("\n\n📝 Тест 3: Обед не дольше 30 минут")
    scenario = matcher.match(
        meal_types=["lunch"],
        people=2,
        max_time_min=30,
        strategy="random"
    )
    
    if scenario:
        print(f"\n✅ {scenario['name']}")
        print(f"   Время: {scenario['estimated_time_min']} мин")
    else:
        print("❌ Нет обедов быстрее 30 минут")
    
    # Тест 4: Получение сценария по ID
    print("\n\n📝 Тест 4: Получение сценария по ID")
    scenario = matcher.get_scenario_by_id("dinner_chicken_vegetables", people=4)
    
    if scenario:
        print(f"\n✅ {scenario['name']} (на {scenario['scaled_for_people']} чел)")
        print(f"\n   Первые 3 ингредиента:")
        for comp in scenario['components'][:3]:
            scaled = comp.get('quantity_scaled')
            print(f"      - {comp['ingredient']}: {scaled}{comp['unit']}")
    
    # Тест 5: Статистика по meal_types
    print("\n\n📝 Тест 5: Все сценарии по типам")
    for meal_type in ["breakfast", "lunch", "dinner", "snack"]:
        scenarios = matcher.get_all_scenarios(meal_type=meal_type)
        print(f"   {meal_type}: {len(scenarios)} сценариев")
        if scenarios:
            names = [s['name'] for s in scenarios[:3]]
            print(f"      Примеры: {', '.join(names)}...")
    
    # Тест 6: Масштабирование для большой группы
    print("\n\n📝 Тест 6: Масштабирование для 10 человек")
    scenario = matcher.match(
        meal_types=["lunch"],
        people=10,
        strategy="simplest"
    )
    
    if scenario:
        print(f"\n✅ {scenario['name']} (на {scenario['scaled_for_people']} чел)")
        print(f"   Коэффициент: x{scenario['scale_factor']:.1f}")
        print(f"\n   Ингредиенты:")
        for comp in scenario['components']:
            scaled = comp.get('quantity_scaled')
            print(f"      - {comp['ingredient']}: {scaled}{comp['unit']}")
    
    print("\n" + "=" * 70)
    print("✅ Тестирование завершено")
    print("=" * 70)


if __name__ == "__main__":
    test_scenario_matcher()
