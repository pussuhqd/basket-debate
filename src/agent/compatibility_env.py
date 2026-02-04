# src/agent/compatibility_env.py
"""
Compatibility Agent Environment - первый агент в Sequential pipeline.

ЦЕЛЬ АГЕНТА:
Формирует КОНЦЕПЦИЮ корзины на основе meal_type (завтрак/обед/ужин).
Обеспечивает ПОЛНОТУ приёма пищи (main_course + side_dish + beverage и т.д.).
НЕ занимается оптимизацией бюджета или тегами - это задача других агентов.

ВЫХОД:
Корзина из 5-12 товаров с покрытием всех required meal_components.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Optional


# ============================================================================
# КЛАСС ОКРУЖЕНИЯ
# ============================================================================

class CompatibilityEnv(gym.Env):
    """
    Gymnasium окружение для Compatibility Agent.
    
    Агент формирует корзину, выбирая товары ПО ОДНОМУ за шаг.
    На каждом шаге агент видит:
    - текущее состояние корзины (observation)
    - какие товары можно добавить (action_mask)
    
    Агент получает reward за:
    - Добавление товаров с недостающими meal_components
    - Разнообразие категорий
    - Приближение к бюджету
    
    Штрафы за:
    - Дубликаты категорий
    - Превышение бюджета
    - Пропуск действий (skip)
    """
    
    metadata = {'render_modes': ['human']}
    
    # ========================================================================
    # ТРЕБОВАНИЯ К ПРИЁМАМ ПИЩИ
    # ========================================================================
    # Это "золотой стандарт" полноценного приёма пищи
    
    MEAL_REQUIREMENTS = {
        'breakfast': {
            # Обязательные компоненты (без них завтрак не завтрак)
            'required': ['bakery', 'beverage'],  # Хлеб + напиток (минимум)
            
            # Опциональные (хорошо бы иметь, но не критично)
            'optional': ['dessert', 'main_course'],  # Сладкое, яйца
            
            'min_items': 3,   # Минимум товаров в корзине
            'max_items': 8    # Максимум (чтобы не раздувать завтрак)
        },
        'lunch': {
            'required': ['main_course', 'side_dish', 'beverage'],  # Полноценный обед
            'optional': ['salad', 'dessert', 'bakery'],  # Салат, десерт, хлеб
            'min_items': 5,
            'max_items': 12
        },
        'dinner': {
            'required': ['main_course', 'side_dish', 'beverage'],  # Как обед
            'optional': ['salad', 'sauce'],  # Салат, соусы
            'min_items': 5,
            'max_items': 12
        },
        'snack': {
            'required': ['beverage'],  # Перекус = хотя бы напиток
            'optional': ['dessert', 'bakery', 'snack'],  # Печенье, снеки
            'min_items': 2,
            'max_items': 5
        }
    }
    
    # ========================================================================
    # ИНИЦИАЛИЗАЦИЯ ОКРУЖЕНИЯ
    # ========================================================================
    
    def __init__(
        self,
        products: List[Dict],     # Список товаров из БД (fetch_candidate_products)
        constraints: Dict,        # Пользовательские ограничения
        max_steps: int = 15       # Максимум шагов в эпизоде
    ):
        """
        Инициализация окружения.
        
        Args:
            products: список словарей с полями:
                - id: int
                - product_name: str
                - product_category: str
                - price_per_unit: float
                - unit: str ('кг', 'л', 'шт')
                - tags: List[str]
                - meal_components: List[str]  ← КРИТИЧНО для CA!
            
            constraints: словарь:
                - budget_rub: float (бюджет в рублях)
                - meal_type: List[str] (например, ['dinner'])
                - people: int
                - exclude_tags: List[str] (игнорируется CA)
                - include_tags: List[str] (игнорируется CA)
            
            max_steps: сколько шагов может сделать агент
        """
        super().__init__()
        
        self.products = products
        self.constraints = constraints
        self.max_steps = max_steps
        self._budget = float(constraints.get("budget_rub", 1500))
        
        self.n_products = len(products)
        
        if self.n_products == 0:
            raise ValueError("❌ Нет доступных товаров!")
        
        # ====================================================================
        # ACTION SPACE: Дискретные действия
        # ====================================================================
        # Агент может выбрать:
        # - индекс товара [0, n_products-1] → добавить товар в корзину
        # - индекс n_products → skip (пропустить шаг)
        
        self.action_space = spaces.Discrete(self.n_products + 1)
        
        # ====================================================================
        # OBSERVATION SPACE: Вектор из 10 чисел
        # ====================================================================
        # Агент видит "сжатое" представление состояния:
        # [0] budget_ratio       - сколько % бюджета потрачено (0.0 - 2.0)
        # [1] cart_size_ratio    - заполненность корзины (0.0 - 1.0)
        # [2] progress           - прогресс эпизода (0.0 - 1.0)
        # [3] required_coverage  - % выполнения required components (0.0 - 1.0)
        # [4] optional_coverage  - % выполнения optional components (0.0 - 1.0)
        # [5] diversity_ratio    - разнообразие категорий (0.0 - 1.0)
        # [6] required_done_flag - все required выполнены? (0.0 или 1.0)
        # [7] budget_ok_flag     - близко к бюджету? (0.0 или 1.0)
        # [8] min_items_flag     - минимум товаров? (0.0 или 1.0)
        # [9] diversity_ok_flag  - хорошее разнообразие? (0.0 или 1.0)
        
        self.observation_space = spaces.Box(
            low=0, high=2.0, shape=(10,), dtype=np.float32
        )
        
        # Инициализируем состояние
        self.reset()
    
    # ========================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ========================================================================
    
    def _get_meal_requirements(self) -> Dict:
        """
        Получить требования для текущего meal_type.
        
        Returns:
            Словарь с ключами: required, optional, min_items, max_items
        """
        meal_type = self.constraints.get('meal_type', ['lunch'])
        
        # meal_type может быть списком ['dinner'] → берём первый элемент
        if isinstance(meal_type, list):
            meal_type = meal_type[0]
        
        # Возвращаем требования или lunch по умолчанию
        return self.MEAL_REQUIREMENTS.get(meal_type, self.MEAL_REQUIREMENTS['lunch'])
    
    def _calculate_component_coverage(self) -> Dict[str, bool]:
        """
        Вычислить, какие meal_components уже есть в корзине.
        
        Пример:
            Корзина: [Курица (main_course), Макароны (side_dish)]
            Результат: {
                'main_course': True,
                'side_dish': True,
                'beverage': False,
                ...
            }
        
        Returns:
            Словарь {component: True/False}
        """
        # Все возможные компоненты
        components = ['main_course', 'side_dish', 'beverage', 'bakery', 
                     'dessert', 'salad', 'sauce']
        
        # Инициализируем все как False
        coverage = {comp: False for comp in components}
        
        # Проходим по товарам в корзине
        for idx in self.cart_indices:
            product = self.products[idx]
            
            # У каждого товара может быть несколько компонентов
            # Например: ['main_course', 'side_dish']
            for comp in product['meal_components']:
                if comp in coverage:
                    coverage[comp] = True  # Отмечаем как покрытый
        
        return coverage
    
    def _get_obs(self) -> np.ndarray:
        """
        Генерация observation (то, что видит агент).
        
        Преобразует текущее состояние корзины в вектор из 10 чисел.
        Агент НЕ видит сами товары напрямую - только статистику.
        
        Returns:
            np.ndarray shape=(10,) dtype=float32
        """
        # Базовые метрики
        budget_ratio = self.current_sum / self._budget if self._budget > 0 else 0
        cart_size = len(self.cart_indices)
        
        # Покрытие компонентов
        coverage = self._calculate_component_coverage()
        requirements = self._get_meal_requirements()
        
        # Сколько required компонентов покрыто? (0.0 - 1.0)
        # Пример: required=['main_course', 'side_dish', 'beverage']
        #         покрыто 2 из 3 → 0.67
        required_count = len(requirements['required'])
        required_covered_count = sum(
            1 for comp in requirements['required'] if coverage.get(comp, False)
        )
        required_coverage = required_covered_count / required_count if required_count > 0 else 0
        
        # Сколько optional компонентов покрыто?
        optional_count = len(requirements['optional'])
        optional_covered_count = sum(
            1 for comp in requirements['optional'] if coverage.get(comp, False)
        )
        optional_coverage = optional_covered_count / optional_count if optional_count > 0 else 0
        
        # Разнообразие категорий (чтобы не было 5 одинаковых макарон)
        if cart_size > 1:
            categories = [self.products[idx]["product_category"] for idx in self.cart_indices]
            unique_categories = len(set(categories))  # Уникальные категории
            diversity_ratio = unique_categories / cart_size
        else:
            diversity_ratio = 0.0
        
        # Собираем observation вектор
        return np.array([
            budget_ratio,                                          # [0]
            cart_size / self.max_steps,                            # [1]
            self.steps / self.max_steps,                           # [2]
            required_coverage,                                     # [3]
            optional_coverage,                                     # [4]
            diversity_ratio,                                       # [5]
            1.0 if required_coverage >= 1.0 else 0.0,             # [6] флаг
            1.0 if 0.8 <= budget_ratio <= 1.2 else 0.0,          # [7] флаг
            1.0 if cart_size >= requirements['min_items'] else 0.0,  # [8] флаг
            1.0 if diversity_ratio > 0.5 else 0.0                 # [9] флаг
        ], dtype=np.float32)
    
    # ========================================================================
    # REWARD FUNCTION
    # ========================================================================
    
    def _calculate_reward(self, action: int) -> float:
        """
        ✅ УЛУЧШЕНО: Более агрессивные штрафы и бонусы
        """
        reward = 0.0
        requirements = self._get_meal_requirements()
        
        # Skip action → штраф
        if action >= self.n_products:
            return -3.0  # Увеличили штраф с -2.0 до -3.0
        
        added_product = self.products[action]
        
        # ====================================================================
        # 1. BONUS ЗА НЕДОСТАЮЩИЕ REQUIRED COMPONENTS (+10)
        # ====================================================================
        coverage_before = self._calculate_component_coverage()
        
        missing_required = set(requirements['required']) - set(
            comp for comp, covered in coverage_before.items() if covered
        )
        
        for comp in added_product['meal_components']:
            if comp in missing_required:
                reward += 10.0  # Увеличили с 8.0 до 10.0
                break
        
        # ====================================================================
        # 2. BONUS ЗА OPTIONAL COMPONENTS (+4)
        # ====================================================================
        missing_optional = set(requirements['optional']) - set(
            comp for comp, covered in coverage_before.items() if covered
        )
        
        for comp in added_product['meal_components']:
            if comp in missing_optional:
                reward += 4.0  # Увеличили с 3.0 до 4.0
                break
        
        # ====================================================================
        # 3. ШТРАФ ЗА ДУБЛИКАТ КАТЕГОРИИ (-8)
        # ====================================================================
        cart_categories = [
            self.products[idx]['product_category'] 
            for idx in self.cart_indices[:-1]
        ]
        
        if added_product['product_category'] in cart_categories:
            reward -= 8.0  # Увеличили с -4.0 до -8.0
        
        # ====================================================================
        # 4. ✅ НОВОЕ: ШТРАФ ЗА НЕПОДХОДЯЩИЕ ТОВАРЫ НА УЖИН/ОБЕД (-6)
        # ====================================================================
        meal_type = self.constraints.get('meal_type', ['lunch'])
        if isinstance(meal_type, list):
            meal_type = meal_type[0]
        
        # Если это обед/ужин, штрафуем за десерты как основное блюдо
        if meal_type in ['lunch', 'dinner']:
            product_name_lower = added_product['product_name'].lower()
            
            # Список неподходящих слов для обеда/ужина
            inappropriate_keywords = [
                'сырок', 'глазированный', 'десерт', 'мороженое', 'конфеты',
                'шоколад', 'печенье', 'торт', 'пирожное', 'зефир'
            ]
            
            for keyword in inappropriate_keywords:
                if keyword in product_name_lower:
                    # Если это первый товар в корзине (должно быть main_course)
                    if len(self.cart_indices) <= 2:
                        reward -= 6.0
                    break
        
        # ====================================================================
        # 5. ШТРАФ ЗА ПРЕВЫШЕНИЕ БЮДЖЕТА
        # ====================================================================
        if self.current_sum > self._budget * 1.2:
            reward -= 12.0  # Увеличили с -10.0
        elif self.current_sum > self._budget:
            reward -= 6.0   # Увеличили с -5.0
        
        # ====================================================================
        # 6. BONUS ЗА ПРИБЛИЖЕНИЕ К БЮДЖЕТУ (+6)
        # ====================================================================
        budget_ratio = self.current_sum / self._budget
        
        if 0.8 <= budget_ratio <= 1.0:
            reward += 6.0  # Увеличили с 5.0
        elif 0.6 <= budget_ratio <= 1.2:
            reward += 3.0  # Увеличили с 2.0
        
        return reward
        
    # ========================================================================
    # ACTION MASKING
    # ========================================================================
    
    def action_masks(self) -> np.ndarray:
        """
        Генерация масок для недопустимых действий.
        
        ЗАЧЕМ ЭТО НУЖНО:
        Без масок агент может:
        - Добавить один товар дважды
        - Выбрать товар, который взорвёт бюджет
        - Потратить шаги на бесполезные действия
        
        С масками агент сфокусирован на ВАЛИДНЫХ действиях.
        
        Returns:
            np.ndarray shape=(n_products+1,) dtype=bool
            True = действие разрешено
            False = действие запрещено (агент не может его выбрать)
        """
        mask = np.ones(self.n_products + 1, dtype=bool)  # Всё разрешено по умолчанию
        
        # ====================================================================
        # 1. МАСКИРОВАТЬ УЖЕ ДОБАВЛЕННЫЕ ТОВАРЫ
        # ====================================================================
        # Если товар с индексом 5 уже в корзине → mask[5] = False
        for idx in self.cart_indices:
            mask[idx] = False
        
        # ====================================================================
        # 2. МАСКИРОВАТЬ ТОВАРЫ, ПРЕВЫШАЮЩИЕ БЮДЖЕТ
        # ====================================================================
        # Если добавление товара приведёт к превышению бюджета на 30%
        for idx in range(self.n_products):
            if mask[idx]:  # Только если товар ещё не замаскирован
                if self.current_sum + self.products[idx]['price_per_unit'] > self._budget * 1.3:
                    mask[idx] = False
        
        # ====================================================================
        # 3. ПРИОРИТИЗАЦИЯ ТОВАРОВ С НЕДОСТАЮЩИМИ COMPONENTS
        # ====================================================================
        # ЛОГИКА: Если корзине не хватает required components,
        #         маскируем товары БЕЗ этих components (чтобы агент сфокусировался)
        
        coverage = self._calculate_component_coverage()
        requirements = self._get_meal_requirements()
        
        # Какие required компоненты ещё не покрыты?
        missing_required = set(requirements['required']) - set(
            comp for comp, covered in coverage.items() if covered
        )
        
        # Если есть недостающие required И корзина не переполнена
        if missing_required and len(self.cart_indices) < requirements['max_items'] - 2:
            # Создаём маску товаров, которые покрывают missing_required
            has_required_mask = np.zeros(self.n_products + 1, dtype=bool)
            
            for idx in range(self.n_products):
                if mask[idx]:  # Только если товар не был замаскирован ранее
                    product_comps = set(self.products[idx]['meal_components'])
                    
                    # Если товар содержит хотя бы один missing_required component
                    if product_comps & missing_required:  # Пересечение множеств
                        has_required_mask[idx] = True
            
            # Если есть товары с required components, приоритизируем их
            if has_required_mask[:self.n_products].any():
                mask[:self.n_products] = has_required_mask[:self.n_products]
        
        # ====================================================================
        # 4. SKIP ВСЕГДА ДОСТУПЕН
        # ====================================================================
        mask[-1] = True  # Последний индекс = skip action
        
        return mask
    
    # ========================================================================
    # GYMNASIUM API: reset() и step()
    # ========================================================================
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Сброс окружения в начальное состояние.
        
        Вызывается в начале каждого эпизода обучения.
        
        Returns:
            observation: np.ndarray shape=(10,)
            info: dict (может содержать доп. информацию)
        """
        super().reset(seed=seed)
        
        self.cart_indices = []     # Пустая корзина
        self.current_sum = 0.0     # Нулевая стоимость
        self.steps = 0             # Счётчик шагов
        
        return self._get_obs(), {}
    
    def step(self, action: int):
        """
        Выполнение действия агента.
        
        Args:
            action: индекс товара (0 - n_products-1) или skip (n_products)
        
        Returns:
            observation: следующее состояние
            reward: награда за действие
            terminated: эпизод завершён успешно?
            truncated: эпизод прерван (лимит шагов)?
            info: дополнительная информация
        """
        self.steps += 1
        
        # ====================================================================
        # ОБРАБОТКА ДЕЙСТВИЯ
        # ====================================================================
        
        if action >= self.n_products:
            # Skip action
            reward = -2.0
        else:
            # Добавление товара
            if action not in self.cart_indices:  # Защита от дубликатов
                self.cart_indices.append(action)
                self.current_sum += self.products[action]['price_per_unit']
                reward = self._calculate_reward(action)
            else:
                # Если каким-то образом попытались добавить дубликат
                reward = -5.0
        
        # ====================================================================
        # СЛЕДУЮЩЕЕ СОСТОЯНИЕ
        # ====================================================================
        obs = self._get_obs()
        
        # ====================================================================
        # УСЛОВИЯ ЗАВЕРШЕНИЯ ЭПИЗОДА
        # ====================================================================
        requirements = self._get_meal_requirements()
        coverage = self._calculate_component_coverage()
        
        # Все required components покрыты?
        required_covered = all(
            coverage.get(comp, False) for comp in requirements['required']
        )
        
        # TERMINATED = эпизод завершён (успешно или нет)
        terminated = (
            len(self.cart_indices) >= requirements['max_items'] or  # Корзина переполнена
            self.current_sum > self._budget * 1.3 or                # Бюджет превышен
            (required_covered and len(self.cart_indices) >= requirements['min_items'])  # Цель достигнута!
        )
        
        # TRUNCATED = эпизод прерван из-за лимита шагов
        truncated = self.steps >= self.max_steps
        
        # ====================================================================
        # ФИНАЛЬНЫЙ REWARD (бонус в конце эпизода)
        # ====================================================================
        if terminated or truncated:
            if required_covered and len(self.cart_indices) >= requirements['min_items']:
                # УСПЕШНАЯ КОРЗИНА → большой бонус!
                reward += 30.0
                
                # Дополнительный бонус за разнообразие категорий
                categories = [self.products[idx]['product_category'] for idx in self.cart_indices]
                unique_categories = len(set(categories))
                
                if unique_categories >= 5:
                    reward += 20.0  # Отличное разнообразие
                elif unique_categories >= 3:
                    reward += 10.0  # Хорошее разнообразие
            
            elif len(self.cart_indices) == 0:
                # ПУСТАЯ КОРЗИНА → штраф
                reward -= 20.0
        
        # ====================================================================
        # INFO СЛОВАРЬ
        # ====================================================================
        info = {
            'cart': [self.products[idx] for idx in self.cart_indices],
            'cart_size': len(self.cart_indices),
            'total_cost': self.current_sum,
            'component_coverage': coverage
        }
        
        return obs, reward, terminated, truncated, info
    
    # ========================================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ========================================================================
    
    def render(self):
        """
        Отображение текущего состояния корзины (для отладки).
        
        Вызывается вручную или при render_mode='human'.
        """
        if len(self.cart_indices) == 0:
            print("🛒 Корзина пуста")
            return
        
        print(f"\n🛒 Корзина ({len(self.cart_indices)} товаров, {self.current_sum:.2f}₽/{self._budget}₽):")
        
        for idx in self.cart_indices:
            product = self.products[idx]
            comps = ', '.join(product['meal_components'])
            print(f"  • {product['product_name'][:50]} - {product['price_per_unit']:.2f}₽ [{comps}]")
        
        # Показываем покрытие компонентов
        coverage = self._calculate_component_coverage()
        requirements = self._get_meal_requirements()
        
        meal_type = self.constraints.get('meal_type', ['lunch'])
        if isinstance(meal_type, list):
            meal_type = meal_type[0]
        
        print(f"\n📊 Покрытие компонентов (meal_type={meal_type}):")
        print("  Обязательные:")
        for comp in requirements['required']:
            status = "✅" if coverage.get(comp, False) else "❌"
            print(f"    {status} {comp}")
        
        print("  Опциональные:")
        for comp in requirements['optional']:
            status = "✅" if coverage.get(comp, False) else "☑️"
            print(f"    {status} {comp}")
    
    def get_cart(self) -> List[Dict]:
        """
        Получить текущую корзину для передачи следующему агенту.
        
        ИСПОЛЬЗОВАНИЕ:
        После завершения работы Compatibility Agent,
        Budget Agent получит эту корзину как входные данные.
        
        Returns:
            List[Dict]: список товаров в корзине
        """
        return [self.products[idx] for idx in self.cart_indices]

