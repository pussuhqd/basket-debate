# src/agent/multi_action_masked_env.py
"""
Multi-Action Environment с Action Masking для специализации ролей.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import Counter


class MultiActionMaskedBasketEnv(gym.Env):
    """
    Multi-Action Environment с ролевой специализацией через маски:
    
    - action[0] (budget_agent): выбирает из ДЕШЁВЫХ товаров
    - action[1] (compat_agent): выбирает из товаров РАЗНЫХ категорий
    - action[2] (profile_agent): выбирает по ТЕГАМ (exclude/include)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "name": "multi_action_masked_v0"}
    
    def __init__(self, products, constraints, max_steps=10, render_mode=None):
        super().__init__()
        
        self.products = products
        self.constraints = constraints
        self._budget = float(constraints.get("budget_rub", 1500))
        self.exclude_tags = constraints.get("exclude_tags", [])
        self.include_tags = constraints.get("include_tags", [])
        self._max_steps = max_steps
        self.render_mode = render_mode
        
        # Action space: 3 независимых выбора
        self.action_space = spaces.MultiDiscrete([len(products) + 1] * 3)
        
        # Observation space: 8 фич
        self.observation_space = spaces.Box(
            low=0, high=2.0, shape=(8,), dtype=np.float32
        )
        
        # Предвычисляем маски для каждой роли
        self._precompute_role_indices()
        
        self.reset()
    
    def _precompute_role_indices(self):
        """
        Предвычисляет индексы товаров для каждой роли:
        - budget_indices: дешёвые товары (price < средняя цена)
        - profile_indices: товары, соответствующие тегам
        """
        # 1. BUDGET AGENT: Дешёвые товары (цена ниже среднего)
        prices = [p["price_per_unit"] for p in self.products if p["price_per_unit"] > 0]
        avg_price = np.mean(prices) if prices else 100
        
        self.budget_indices = [
            i for i, p in enumerate(self.products)
            if p["price_per_unit"] > 0 and p["price_per_unit"] <= avg_price
        ]
        
        # 2. PROFILE AGENT: Товары без exclude_tags
        self.profile_indices = []
        for i, p in enumerate(self.products):
            if p["price_per_unit"] == 0:
                continue
            
            p_tags = set(p["tags"])
            
            # Исключаем товары с запрещёнными тегами
            if self.exclude_tags and (p_tags & set(self.exclude_tags)):
                continue
            
            # Если есть include_tags, берём только с нужными тегами
            if self.include_tags:
                if p_tags & set(self.include_tags):
                    self.profile_indices.append(i)
            else:
                self.profile_indices.append(i)
        
        # Резервные индексы (если фильтры слишком строгие)
        all_valid_indices = [
            i for i, p in enumerate(self.products) if p["price_per_unit"] > 0
        ]
        
        if not self.budget_indices:
            self.budget_indices = all_valid_indices[:50]  # Первые 50
        
        if not self.profile_indices:
            self.profile_indices = all_valid_indices
        
        print(f"[DEBUG] Budget indices: {len(self.budget_indices)}")
        print(f"[DEBUG] Profile indices: {len(self.profile_indices)}")
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cart = []
        self.current_sum = 0.0
        self.steps = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        budget_ratio = self.current_sum / self._budget if self._budget > 0 else 0
        cart_size = len(self.cart)
        
        # Разнообразие категорий
        if cart_size > 1:
            categories = [self.products[idx]["product_category"] for idx in self.cart]
            diversity_ratio = len(set(categories)) / cart_size
        else:
            diversity_ratio = 0.0
        
        # Средняя цена товара
        if cart_size > 0:
            avg_price = self.current_sum / cart_size
            avg_price_ratio = avg_price / (self._budget / 10) if self._budget > 0 else 0
        else:
            avg_price_ratio = 0.0
        
        return np.array([
            budget_ratio,
            cart_size / self._max_steps,
            self.steps / self._max_steps,
            diversity_ratio,
            avg_price_ratio,
            1.0 if cart_size > 0 else 0.0,
            1.0 if budget_ratio > 0.8 else 0.0,
            1.0 if diversity_ratio > 0.5 else 0.0
        ], dtype=np.float32)
    
    def action_masks(self):
        """
        Возвращает маски для каждой роли (True = действие разрешено).
        
        Returns:
            List[np.ndarray]: 3 маски (по одной на каждое действие)
        """
        num_products = len(self.products)
        
        # === МАСКА 1: BUDGET AGENT (дешёвые товары) ===
        budget_mask = np.zeros(num_products + 1, dtype=bool)
        budget_mask[self.budget_indices] = True
        budget_mask[-1] = True  # skip всегда доступен
        
        # === МАСКА 2: COMPAT AGENT (разнообразие категорий) ===
        compat_mask = np.zeros(num_products + 1, dtype=bool)
        
        if len(self.cart) == 0:
            # Если корзина пустая, выбираем из всех
            compat_mask[:num_products] = True
        else:
            # Выбираем товары ДРУГИХ категорий
            cart_categories = set(
                self.products[idx]["product_category"] for idx in self.cart
            )
            
            for i, p in enumerate(self.products):
                if p["price_per_unit"] > 0:
                    if p["product_category"] not in cart_categories:
                        compat_mask[i] = True
            
            # Если все категории уже есть, разрешаем всё
            if not compat_mask[:num_products].any():
                compat_mask[:num_products] = True
        
        compat_mask[-1] = True  # skip
        
        # === МАСКА 3: PROFILE AGENT (теги) ===
        profile_mask = np.zeros(num_products + 1, dtype=bool)
        profile_mask[self.profile_indices] = True
        profile_mask[-1] = True  # skip
        
        return [budget_mask, compat_mask, profile_mask]
    
    def step(self, action):
        """
        action = [budget_action, compat_action, profile_action]
        """
        self.steps += 1
        reward = 0.0
        
        # Обрабатываем 3 действия
        added_count = 0
        unique_actions = len(set(action))
        
        for act in action:
            if act < len(self.products):
                product = self.products[act]
                price = product["price_per_unit"]
                
                # НОВОЕ: Проверка на дубликаты (не добавляем, если товар уже в корзине)
                if act in self.cart:
                    continue  # Пропускаем этот товар
                
                if self.current_sum + price <= self._budget * 1.2:
                    self.cart.append(act)
                    self.current_sum += price
                    added_count += 1

        
        # === REWARDS ===
        
        # Бонус за добавление товаров
        reward += added_count * 3.0
        
        # Бонус за разные действия (специализация работает!)
        if unique_actions == 3:
            reward += 5.0  # Увеличили с 3.0 (поощряем специализацию)
        elif unique_actions == 2:
            reward += 2.0
        
        # Штраф за skip всех
        if added_count == 0:
            reward -= 5.0
        
        # Бонус за бюджет
        budget_ratio = self.current_sum / self._budget if self._budget > 0 else 0
        if 0.8 <= budget_ratio <= 1.2:
            reward += 10.0  # Увеличили с 8.0
        elif 0.6 <= budget_ratio <= 1.3:
            reward += 5.0
        
        # Бонус за разнообразие
        cart_size = len(self.cart)
        if cart_size > 1:
            categories = [self.products[idx]["product_category"] for idx in self.cart]
            unique_categories = len(set(categories))
            reward += unique_categories * 2.0  # Увеличили с 1.5
        
        # Штраф за дубликаты
        if cart_size > 2:
            product_counts = Counter(self.cart)
            for count in product_counts.values():
                if count > 2:
                    reward -= 1.5 * (count - 2)
        
        # Финальный reward
        done = self.steps >= self._max_steps
        
        if done:
            if cart_size >= 5 and 0.7 <= budget_ratio <= 1.2:
                categories = [self.products[idx]["product_category"] for idx in self.cart]
                unique_categories = len(set(categories))
                if unique_categories >= 5:
                    reward += 40.0  # Увеличили с 30.0
                elif unique_categories >= 3:
                    reward += 25.0
                else:
                    reward += 10.0
            elif cart_size == 0:
                reward -= 30.0
            elif cart_size < 5:
                reward -= 10.0
        
        return self._get_obs(), reward, done, False, {
            "cart_size": cart_size,
            "budget_ratio": budget_ratio,
            "cart_sum": self.current_sum
        }
    
    def render(self):
        if self.render_mode == "human":
            print(f"Step {self.steps}: Cart={len(self.cart)}, Sum={self.current_sum:.2f}₽")


def create_masked_env(products, constraints, max_steps=10, render_mode=None):
    """Factory function."""
    return MultiActionMaskedBasketEnv(products, constraints, max_steps, render_mode)
