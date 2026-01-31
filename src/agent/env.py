# src/agent/env.py
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv


class BasketEnv(ParallelEnv):
    """
    MAS для корзины: 3 агента оптимизируют по запросу.
    
    Агенты:
    - budget_agent: минимизирует отклонение от бюджета
    - compat_agent: проверяет совместимость товаров
    - profile_agent: учитывает предпочтения пользователя
    """
    metadata = {"render_modes": ["human"], "name": "basket_mas_v0"}
    
    def __init__(
        self, 
        products,  
        constraints,  
        max_steps=10
    ):
        """
        Args:
            budget: Бюджет в рублях
            max_steps: Максимальное количество шагов симуляции
            exclude_tags: Список запрещённых тегов (например, ['dairy'])
            include_tags: Список обязательных тегов (например, ['vegan'])
            meal_type: Тип приёма пищи (['breakfast', 'lunch', 'dinner', 'snack'])
            people: Количество человек
        """
        super().__init__()
        
        self.products = products  
        self.constraints = constraints  
        self._max_steps = int(max_steps)

        # Извлекаем параметры из constraints
        self._budget = float(constraints.get("budget_rub", 1500))
        self.exclude_tags = constraints.get("exclude_tags", [])
        self.include_tags = constraints.get("include_tags", [])
        self.meal_type = constraints.get("meal_type", [])
        self.people = constraints.get("people", 1)

        # Агенты
        self.possible_agents = ["budget_agent", "compat_agent", "profile_agent"]
        self.agents = self.possible_agents.copy()
        
        # Пространства наблюдений и действий
        self._observation_spaces = {
            agent: spaces.Box(low=0, high=2000, shape=(12,), dtype=np.float32) 
            for agent in self.possible_agents
        }

        num_actions = len(products) + 1  # +1 для действия "skip"
        self._action_spaces = {
            agent: spaces.Discrete(num_actions)
            for agent in self.possible_agents
        }
        
        # Состояние корзины
        self.current_sum = 0.0
        self.cart = []
        self.steps = 0
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
    
    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def reset(self, seed=None, options=None):
        """Сброс окружения."""
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents.copy()
        self.current_sum = 0.0
        self.cart = []
        self.steps = 0
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        
        # Формируем начальные наблюдения для каждого агента
        obs_dict = {}
        for agent in self.agents:
            obs_array = np.array(
                [self._budget, 0.0] + [0.0] * 10,
                dtype=np.float32
            )
            obs_dict[agent] = obs_array
        
        # Мета-информация (доступна агентам)
        infos = {
            agent: {
                "budget": self._budget,
                "exclude_tags": self.exclude_tags,
                "include_tags": self.include_tags,
                "meal_type": self.meal_type,
                "people": self.people
            } 
            for agent in self.agents
        }
        
        return obs_dict, infos
    
    def step(self, actions):
        """
        Шаг симуляции. Каждый агент выбирает действие (добавить товар или пропустить).
        
        Args:
            actions: dict {agent_name: action_id}
        
        Returns:
            observations, rewards, terminations, truncations, infos
        """
        # Обрабатываем действия агентов
        for agent, action in actions.items():
            if action > 0:  # 0 = skip, 1..N = выбрать товар
                product_idx = action - 1  # action=1 → индекс 0 (первый товар)
                
                # Проверяем, что индекс в пределах списка
                if product_idx < len(self.products):
                    product = self.products[product_idx]  # Словарь с товаром
                    price = product["price_per_unit"]  # Реальная цена из БД
                    
                    # Проверяем, не превысим ли бюджет (допускаем +10%)
                    if self.current_sum + price <= self._budget * 1.1:
                        # Добавляем в корзину ИНДЕКС товара (не цену!)
                        # Потом по индексу достанем полную информацию
                        self.cart.append(product_idx)
                        self.current_sum += price

        
        self.steps += 1
        
        # === РАСЧЁТ НАГРАД (Reward Function) ===
        rewards = {agent: 0.0 for agent in self.agents}

        # 1. BUDGET AGENT
        budget_diff = abs(self.current_sum - self._budget)
        budget_ratio = self.current_sum / self._budget if self._budget > 0 else 0

        # Штраф за отклонение от бюджета
        rewards["budget_agent"] -= budget_diff / self._budget

        # Бонус за близость к целевому бюджету (90%-100% = идеально)
        if 0.9 <= budget_ratio <= 1.0:
            rewards["budget_agent"] += 5.0  # Сильный бонус за точность

        # 2. COMPATIBILITY AGENT
        if len(self.cart) > 1:
            # Считаем уникальные категории товаров
            categories = [self.products[idx]["product_category"] for idx in self.cart]
            unique_categories = len(set(categories))
            diversity_ratio = unique_categories / len(self.cart)
            
            # Бонус за разнообразие
            rewards["compat_agent"] += diversity_ratio * 3.0
            
            # ЖЁСТКИЙ ШТРАФ за однообразие (например, 10 товаров воды)
            if unique_categories == 1 and len(self.cart) >= 3:
                rewards["compat_agent"] -= 10.0  # Все товары из одной категории!
            
            # НОВОЕ: Штраф за слишком много одинаковых товаров
            from collections import Counter
            product_counts = Counter(self.cart)
            for product_idx, count in product_counts.items():
                if count > 2:  # Один товар больше 2 раз
                    rewards["compat_agent"] -= 3.0 * (count - 2)  # -3 за каждый лишний


                # 3. PROFILE AGENT
                for idx in self.cart:
                    product = self.products[idx]
                    product_tags = set(product["tags"])
                    
                    # ЖЁСТКИЙ ШТРАФ за нарушение exclude_tags
                    if product_tags & set(self.exclude_tags):  # Пересечение множеств
                        rewards["profile_agent"] -= 5.0  # За каждый запрещённый товар
                    
                    # Бонус за соответствие include_tags
                    if self.include_tags and product_tags & set(self.include_tags):
                        rewards["profile_agent"] += 2.0
                
        # Обновляем накопленные награды
        for agent, r in rewards.items():
            self._cumulative_rewards[agent] += r
        
        # === НОВЫЕ НАБЛЮДЕНИЯ ===
        obs = {}
        for agent in self.agents:
            obs_array = np.array(
                [
                    self._budget - self.current_sum,  # остаток бюджета
                    self.current_sum,                 # текущая сумма корзины
                ] + (np.random.rand(10) * 100).tolist(),  # TODO: реальные фичи товаров
                dtype=np.float32
            )
            obs[agent] = obs_array
        
        # === ПРОВЕРКА ЗАВЕРШЕНИЯ ===
        done = self.steps >= self._max_steps
        terms = {agent: done for agent in self.agents}
        truncs = {agent: False for agent in self.agents}
        
        infos = {
            agent: {
                "cart_sum": self.current_sum, 
                "cart_size": len(self.cart),
                "cumulative_reward": self._cumulative_rewards[agent]
            } 
            for agent in self.agents
        }
        
        # Если эпизод завершён, убираем агентов
        if done:
            self.agents = []
        
        return obs, rewards, terms, truncs, infos
    
    def render(self):
        """Вывод текущего состояния (для дебага)."""
        print(f"Step {self.steps}: Cart={self.cart}, Sum={self.current_sum:.2f}")
    
    def close(self):
        """Очистка ресурсов."""
        pass


def create_basket_env(products, constraints, max_steps=10):
    """
    Factory-функция для создания окружения.
    
    Args:
        products: список товаров из БД (каждый = словарь)
        constraints: словарь с параметрами запроса (budget_rub, exclude_tags, ...)
        max_steps: количество шагов симуляции
    
    Returns:
        BasketEnv: окружение с реальными товарами
    """
    return BasketEnv(
        products=products,
        constraints=constraints,
        max_steps=int(max_steps)
    )
