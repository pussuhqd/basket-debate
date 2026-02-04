"""
Модуль для выбора сценария блюда на основе запроса пользователя.

Основные функции:
- Загрузка сценариев из scenarios.json
- Фильтрация по meal_type, времени приготовления, количеству порций
- Выбор оптимального сценария (случайный или по приоритету)
- Масштабирование ингредиентов под количество людей

Использование:
    matcher = ScenarioMatcher()
    scenario = matcher.match(
        meal_types=["dinner"],
        people=3,
        max_time_min=30
    )
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import random
from copy import deepcopy


# ==================== КОНФИГУРАЦИЯ ====================

SCENARIOS_PATH = Path("data/scenarios.json")


# ==================== КЛАСС ScenarioMatcher ====================

class ScenarioMatcher:
    """
    Класс для выбора сценария блюда из библиотеки сценариев.
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
        """
        Загружает сценарии из JSON файла.
        """
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
        
        # Выводим статистику по meal_types
        meal_type_counts = {}
        for scenario in self.scenarios:
            meal_type = scenario.get('meal_type', 'unknown')
            meal_type_counts[meal_type] = meal_type_counts.get(meal_type, 0) + 1
        
        print(f"   Распределение по типам:")
        for meal_type, count in sorted(meal_type_counts.items()):
            print(f"     - {meal_type}: {count}")
    
    
    def _filter_scenarios(
        self,
        meal_types: Optional[List[str]] = None,
        max_time_min: Optional[int] = None,
        min_serves: Optional[int] = None
    ) -> List[Dict]:
        """
        Фильтрует сценарии по заданным критериям.
        
        Args:
            meal_types: Список типов приемов пищи (breakfast, lunch, dinner, snack)
            max_time_min: Максимальное время приготовления в минутах
            min_serves: Минимальное базовое количество порций
        
        Returns:
            List[Dict]: Отфильтрованные сценарии
        """
        filtered = self.scenarios.copy()
        
        # Фильтр по meal_type
        if meal_types:
            filtered = [
                s for s in filtered
                if s.get('meal_type') in meal_types
            ]
        
        # Фильтр по времени приготовления
        if max_time_min is not None:
            filtered = [
                s for s in filtered
                if s.get('estimated_time_min', 999) <= max_time_min
            ]
        
        # Фильтр по базовому количеству порций
        if min_serves is not None:
            filtered = [
                s for s in filtered
                if s.get('serves_base', 1) >= min_serves
            ]
        
        return filtered
    
    
    def _scale_scenario(self, scenario: Dict, people: int) -> Dict:
        """
        Масштабирует количество ингредиентов в сценарии под количество людей.
        
        Args:
            scenario: Исходный сценарий
            people: Количество человек
        
        Returns:
            Dict: Сценарий с пересчитанными количествами
        """
        # Создаем глубокую копию, чтобы не изменять оригинал
        scaled_scenario = deepcopy(scenario)
        
        serves_base = scenario.get('serves_base', 1)
        scale_factor = people / serves_base
        
        # Масштабируем каждый ингредиент
        for component in scaled_scenario.get('components', []):
            original_quantity = component['quantity_per_person']
            
            # Пересчитываем количество
            # Округляем до разумных значений
            scaled_quantity = original_quantity * scale_factor
            
            # Округление в зависимости от величины
            if scaled_quantity < 10:
                # Для маленьких значений (специи) - до целых
                scaled_quantity = round(scaled_quantity)
            elif scaled_quantity < 100:
                # Для средних значений - до 5г/мл
                scaled_quantity = round(scaled_quantity / 5) * 5
            else:
                # Для больших значений - до 10г/мл
                scaled_quantity = round(scaled_quantity / 10) * 10
            
            component['quantity_scaled'] = max(scaled_quantity, 1)  # Минимум 1
        
        scaled_scenario['scaled_for_people'] = people
        scaled_scenario['scale_factor'] = scale_factor
        
        return scaled_scenario
    
    
    def match(
        self,
        meal_types: Optional[List[str]] = None,
        people: int = 1,
        max_time_min: Optional[int] = None,
        strategy: str = "random"
    ) -> Optional[Dict]:
        """
        Выбирает подходящий сценарий на основе критериев.
        
        Args:
            meal_types: Типы приемов пищи (например, ["dinner"])
            people: Количество человек
            max_time_min: Максимальное время приготовления
            strategy: Стратегия выбора:
                - "random" - случайный из подходящих
                - "fastest" - самый быстрый
                - "simplest" - с минимумом ингредиентов
                - "first" - первый подходящий
        
        Returns:
            Dict: Выбранный сценарий с масштабированными количествами
                  или None если не найдено подходящих
        """
        # 1. Фильтруем сценарии
        candidates = self._filter_scenarios(
            meal_types=meal_types,
            max_time_min=max_time_min
        )
        
        if not candidates:
            print(f"⚠️  Не найдено сценариев для meal_types={meal_types}, max_time={max_time_min}")
            return None
        
        # 2. Выбираем сценарий по стратегии
        if strategy == "random":
            selected = random.choice(candidates)
        
        elif strategy == "fastest":
            selected = min(candidates, key=lambda s: s.get('estimated_time_min', 999))
        
        elif strategy == "simplest":
            selected = min(candidates, key=lambda s: len(s.get('components', [])))
        
        elif strategy == "first":
            selected = candidates[0]
        
        else:
            print(f"⚠️  Неизвестная стратегия '{strategy}', используется 'random'")
            selected = random.choice(candidates)
        
        # 3. Масштабируем под количество людей
        scaled_scenario = self._scale_scenario(selected, people)
        
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
