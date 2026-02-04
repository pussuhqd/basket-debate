# src/rl/train_shared_ppo.py
"""
Финальный пайплайн: Multi-Action Agent (замена MAS).
"""
from stable_baselines3 import PPO
from src.backend.db.queries import fetch_candidate_products
from src.agent.multi_action_env import create_multi_action_env
from src.agent.utils import pad_products_to_k

K = 100

def make_env(seed=0):
    constraints = {
        "budget_rub": 1500,
        "exclude_tags": [],
        "include_tags": [],
        "meal_type": ["dinner"],
        "people": 3,
    }
    
    products = fetch_candidate_products(constraints, limit=K)
    products = pad_products_to_k(products, k=K)
    print(f"[INFO] Fetched {len(products)} products")
    
    return create_multi_action_env(products, constraints, max_steps=10)

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Обучение Multi-Action Agent")
    print("=" * 60)
    
    env = make_env()
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=128,
        ent_coef=0.05,
        clip_range=0.3,
        gamma=0.99,
        tensorboard_log="./logs/ppo_multi_action/"
    )
    
    print("\n🚀 Обучаем 200k шагов...")
    model.learn(total_timesteps=200_000)
    
    model.save("models/ppo_shared_v0")
    print("✅ Модель сохранена: models/ppo_shared_v0.zip")
