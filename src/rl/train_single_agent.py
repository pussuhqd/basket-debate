# src/rl/train_single_agent.py
"""
УПРОЩЕННАЯ ВЕРСИЯ: Один агент выбирает все товары.
"""
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from gymnasium import spaces

from src.backend.db.queries import fetch_candidate_products
from src.agent.utils import pad_products_to_k

K = 100

class SingleAgentBasketEnv(gym.Env):
    """Упрощенная версия: один агент, одно действие за шаг."""
    
    def __init__(self, products, constraints, max_steps=10):
        super().__init__()
        
        self.products = products
        self.constraints = constraints
        self._budget = float(constraints.get("budget_rub", 1500))
        self.exclude_tags = constraints.get("exclude_tags", [])
        self._max_steps = max_steps
        
        # Action space: выбрать товар 0..K-1 или skip (K)
        self.action_space = spaces.Discrete(len(products) + 1)
        
        # Observation space: простые 6 фич
        self.observation_space = spaces.Box(
            low=0, high=2.0, shape=(6,), dtype=np.float32
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
        
        return np.array([
            budget_ratio,           # 0-1.2
            cart_size / self._max_steps,  # 0-1
            self.steps / self._max_steps, # 0-1
            diversity_ratio,        # 0-1
            1.0 if cart_size > 0 else 0.0,  # Флаг: есть товары
            1.0 if budget_ratio > 0.8 else 0.0  # Флаг: близко к бюджету
        ], dtype=np.float32)
    
    def step(self, action):
        self.steps += 1
        reward = 0.0
        
        # Обработка действия
        if action < len(self.products):  # Выбрать товар
            product = self.products[action]
            price = product["price_per_unit"]
            
            if self.current_sum + price <= self._budget * 1.2:
                self.cart.append(action)
                self.current_sum += price
                
                # ПРОСТОЙ REWARD: Бонус за добавление
                reward += 3.0
        else:
            # Skip: небольшой штраф
            reward -= 1.0
        
        # Бонус за близость к бюджету
        budget_ratio = self.current_sum / self._budget if self._budget > 0 else 0
        if 0.8 <= budget_ratio <= 1.2:
            reward += 5.0
        
        # Бонус за разнообразие
        cart_size = len(self.cart)
        if cart_size > 1:
            categories = [self.products[idx]["product_category"] for idx in self.cart]
            unique_categories = len(set(categories))
            reward += unique_categories * 1.0
        
        # Завершение
        done = self.steps >= self._max_steps
        
        if done:
            if cart_size >= 5 and 0.7 <= budget_ratio <= 1.2:
                categories = [self.products[idx]["product_category"] for idx in self.cart]
                unique_categories = len(set(categories))
                if unique_categories >= 3:
                    reward += 30.0  # Большой финальный бонус
            elif cart_size == 0:
                reward -= 30.0
        
        return self._get_obs(), reward, done, False, {"cart_size": cart_size}
    
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
    
    return SingleAgentBasketEnv(products, constraints, max_steps=10)

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ТЕСТОВОЕ ОБУЧЕНИЕ: Single Agent")
    print("=" * 60)
    
    env = make_env()
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=128,
        ent_coef=0.01,  # Маленькая энтропия для single agent
        clip_range=0.2,
        gamma=0.99
    )
    
    print("\n🚀 Обучаем 100k шагов...")
    model.learn(total_timesteps=100_000)
    
    model.save("models/ppo_single_test")
    print("✅ Модель сохранена: models/ppo_single_test.zip")
    
    # Тестируем
    print("\n🧪 Тестируем 10 эпизодов...")
    obs, _ = env.reset()
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
        print(f"  Эпизод {ep+1}: reward={ep_reward:.1f}, cart_size={info['cart_size']}")
    
    print(f"\n📊 Средний reward: {np.mean(total_rewards):.1f}")
