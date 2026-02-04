# src/agent/multi_action_env.py
"""
Multi-Action Basket Environment (финальная версия).
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import Counter


class MultiActionBasketEnv(gym.Env):
    """
    Агент выбирает 3 товара за шаг (имитация 3 ролей MAS):
    - action[0]: budget-focused (дешёвые)
    - action[1]: compatibility-focused (разнообразие)
    - action[2]: profile-focused (теги)
    """
    
    metadata = {"render_modes": ["human", "ansi"], "name": "multi_action_basket_v0"}
    
    def __init__(self, products, constraints, max_steps=10, render_mode=None):
        super().__init__()
        
        self.products = products
        self.constraints = constraints
        self._budget = float(constraints.get("budget_rub", 1500))
        self.exclude_tags = constraints.get("exclude_tags", [])
        self.include_tags = constraints.get("include_tags", [])
        self._max_steps = max_steps
        self.render_mode = render_mode
        
        # Action space: 3 независимых выбора (товар 0..K-1 или skip K)
        self.action_space = spaces.MultiDiscrete([len(products) + 1] * 3)
        
        # Observation space: 8 фич
        self.observation_space = spaces.Box(
            low=0, high=2.0, shape=(8,), dtype=np.float32
        )
        
        self.reset()
    
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
    
    def step(self, action):
        self.steps += 1
        reward = 0.0
        
        # Обрабатываем 3 действия
        added_count = 0
        unique_actions = len(set(action))
        
        for act in action:
            if act < len(self.products):
                product = self.products[act]
                price = product["price_per_unit"]
                
                if self.current_sum + price <= self._budget * 1.2:
                    self.cart.append(act)
                    self.current_sum += price
                    added_count += 1
        
        # === REWARDS ===
        
        # Бонус за добавление товаров
        reward += added_count * 3.0
        
        # Бонус за разные действия
        if unique_actions == 3:
            reward += 3.0
        elif unique_actions == 2:
            reward += 1.0
        
        # Штраф за skip всех
        if added_count == 0:
            reward -= 5.0
        
        # Бонус за бюджет
        budget_ratio = self.current_sum / self._budget if self._budget > 0 else 0
        if 0.8 <= budget_ratio <= 1.2:
            reward += 8.0
        elif 0.6 <= budget_ratio <= 1.3:
            reward += 4.0
        
        # Бонус за разнообразие
        cart_size = len(self.cart)
        if cart_size > 1:
            categories = [self.products[idx]["product_category"] for idx in self.cart]
            unique_categories = len(set(categories))
            reward += unique_categories * 1.5
        
        # Штраф за дубликаты
        if cart_size > 2:
            product_counts = Counter(self.cart)
            for count in product_counts.values():
                if count > 2:
                    reward -= 1.0 * (count - 2)
        
        # Штраф за нарушение тегов
        if cart_size > 0 and self.exclude_tags:
            violation_count = sum(
                1 for idx in self.cart
                if set(self.products[idx]["tags"]) & set(self.exclude_tags)
            )
            reward -= violation_count * 2.0
        
        # Финальный reward
        done = self.steps >= self._max_steps
        
        if done:
            if cart_size >= 5 and 0.7 <= budget_ratio <= 1.2:
                categories = [self.products[idx]["product_category"] for idx in self.cart]
                unique_categories = len(set(categories))
                if unique_categories >= 3:
                    reward += 30.0
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


def create_multi_action_env(products, constraints, max_steps=10, render_mode=None):
    """Factory function."""
    return MultiActionBasketEnv(products, constraints, max_steps, render_mode)
