# src/rl/train_multi_action.py
"""
Multi-Action Agent: На каждом шаге выбирает 3 товара (имитация MAS).
"""
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces

from src.backend.db.queries import fetch_candidate_products
from src.agent.utils import pad_products_to_k

K = 100

class MultiActionBasketEnv(gym.Env):
    """
    Агент на каждом шаге выбирает 3 действия:
    - action[0]: budget_agent (выбор из дешёвых товаров)
    - action[1]: compat_agent (выбор для разнообразия)
    - action[2]: profile_agent (выбор по тегам)
    """
    
    def __init__(self, products, constraints, max_steps=10):
        super().__init__()
        
        self.products = products
        self.constraints = constraints
        self._budget = float(constraints.get("budget_rub", 1500))
        self.exclude_tags = constraints.get("exclude_tags", [])
        self._max_steps = max_steps
        
        # Action space: 3 действия (каждое = товар 0..K-1 или skip K)
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
            avg_price_ratio = avg_price / (self._budget / 10)
        else:
            avg_price_ratio = 0.0
        
        return np.array([
            budget_ratio,                      # 1
            cart_size / self._max_steps,       # 2
            self.steps / self._max_steps,      # 3
            diversity_ratio,                   # 4
            avg_price_ratio,                   # 5
            1.0 if cart_size > 0 else 0.0,     # 6: флаг наличия товаров
            1.0 if budget_ratio > 0.8 else 0.0,  # 7: флаг близости к бюджету
            1.0 if diversity_ratio > 0.5 else 0.0  # 8: флаг разнообразия
        ], dtype=np.float32)
    
    def step(self, action):
        """
        action = [action_budget, action_compat, action_profile]
        """
        self.steps += 1
        reward = 0.0
        
        # Обрабатываем 3 действия
        added_count = 0
        unique_actions = len(set(action))  # Для бонуса за разные действия
        
        for idx, act in enumerate(action):
            if act < len(self.products):  # Не skip
                product = self.products[act]
                price = product["price_per_unit"]
                
                if self.current_sum + price <= self._budget * 1.2:
                    self.cart.append(act)
                    self.current_sum += price
                    added_count += 1
        
        # REWARD: Бонус за добавление товаров
        reward += added_count * 3.0
        
        # REWARD: Бонус за разные действия (diversity)
        if unique_actions == 3:
            reward += 3.0  # Все 3 действия разные!
        elif unique_actions == 2:
            reward += 1.0
        
        # REWARD: Штраф за skip всех
        if added_count == 0:
            reward -= 5.0
        
        # REWARD: Бонус за близость к бюджету
        budget_ratio = self.current_sum / self._budget if self._budget > 0 else 0
        if 0.8 <= budget_ratio <= 1.2:
            reward += 8.0
        elif 0.6 <= budget_ratio <= 1.3:
            reward += 4.0
        
        # REWARD: Бонус за разнообразие категорий
        cart_size = len(self.cart)
        if cart_size > 1:
            categories = [self.products[idx]["product_category"] for idx in self.cart]
            unique_categories = len(set(categories))
            reward += unique_categories * 1.5
        
        # Штраф за дубликаты
        if cart_size > 2:
            from collections import Counter
            product_counts = Counter(self.cart)
            for count in product_counts.values():
                if count > 2:
                    reward -= 1.0 * (count - 2)
        
        # Завершение
        done = self.steps >= self._max_steps
        
        if done:
            if cart_size >= 5 and 0.7 <= budget_ratio <= 1.2:
                categories = [self.products[idx]["product_category"] for idx in self.cart]
                unique_categories = len(set(categories))
                if unique_categories >= 3:
                    reward += 30.0  # Большой финальный бонус
                else:
                    reward += 10.0
            elif cart_size == 0:
                reward -= 30.0
            elif cart_size < 5:
                reward -= 10.0
        
        return self._get_obs(), reward, done, False, {
            "cart_size": cart_size,
            "budget_ratio": budget_ratio
        }

def make_env():
    constraints = {
        "budget_rub": 1500,
        "exclude_tags": [],
        "include_tags": [],
        "meal_type": ["dinner"],
        "people": 3,
    }
    
    products = fetch_candidate_products(constraints, limit=K)
    products = pad_products_to_k(products, k=K)
    print(f"[INFO] Loaded {len(products)} products")
    
    return MultiActionBasketEnv(products, constraints, max_steps=10)

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Multi-Action Agent (Псевдо-MAS)")
    print("=" * 60)
    
    env = make_env()
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=128,
        ent_coef=0.05,  # Чуть больше для разнообразия действий
        clip_range=0.3,
        gamma=0.99
    )
    
    print("\n🚀 Обучаем 100k шагов...")
    model.learn(total_timesteps=100_000)
    
    model.save("models/ppo_multi_action")
    print("✅ Модель сохранена: models/ppo_multi_action.zip")
    
    # Тестируем
    print("\n🧪 Тестируем 10 эпизодов...")
    total_rewards = []
    
    for ep in range(10):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            ep_reward += reward
        
        total_rewards.append(ep_reward)
        print(f"  Эпизод {ep+1}: reward={ep_reward:.1f}, cart={info['cart_size']}, budget={info['budget_ratio']:.2f}")
    
    print(f"\n📊 Средний reward: {np.mean(total_rewards):.1f}")
