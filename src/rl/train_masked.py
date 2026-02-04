# src/rl/train_masked.py
"""
Обучение Multi-Action Agent с Action Masking.
"""
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.backend.db.queries import fetch_candidate_products
from src.agent.multi_action_masked_env import create_masked_env
from src.agent.utils import pad_products_to_k

K = 100

def mask_fn(env):
    """Обёртка для получения масок из окружения."""
    return env.action_masks()

def make_env(seed=0):
    constraints = {
        "budget_rub": 1500,
        "exclude_tags": ["dairy"],  # Пример: исключаем молочку
        "include_tags": [],
        "meal_type": ["dinner"],
        "people": 3,
    }
    
    products = fetch_candidate_products(constraints, limit=K)
    products = pad_products_to_k(products, k=K)
    print(f"[INFO] Fetched {len(products)} products")
    
    env = create_masked_env(products, constraints, max_steps=10)
    
    # Оборачиваем в ActionMasker (автоматически применяет маски)
    env = ActionMasker(env, mask_fn)
    
    return env

if __name__ == "__main__":
    print("=" * 70)
    print("🎭 Обучение Multi-Action Agent с Action Masking")
    print("=" * 70)
    
    env = make_env()
    
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=5e-4,
        n_steps=2048,
        batch_size=128,
        ent_coef=0.15,
        clip_range=0.3,
        gamma=0.99,
        tensorboard_log="./logs/ppo_masked/"
    )
    
    print("\n🚀 Обучаем 150k шагов...")
    model.learn(total_timesteps=150_000)
    
    model.save("models/ppo_masked_v0")
    print("✅ Модель сохранена: models/ppo_masked_v0.zip")
    
    # Тестируем
    print("\n🧪 Тестируем 10 эпизодов...")
    total_rewards = []
    
    for ep in range(10):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        actions_log = []
        
        while not done:
            # MaskablePPO автоматически учитывает маски
            action, _ = model.predict(obs, deterministic=True)
            actions_log.append(action)
            obs, reward, done, _, info = env.step(action)
            ep_reward += reward
        
        total_rewards.append(ep_reward)
        
        # Проверяем разнообразие действий
        unique_budget = len(set([a[0] for a in actions_log]))
        unique_compat = len(set([a[1] for a in actions_log]))
        unique_profile = len(set([a[2] for a in actions_log]))
        
        print(f"  Эпизод {ep+1}:")
        print(f"    reward={ep_reward:.1f}, cart={info['cart_size']}, budget={info['budget_ratio']:.2f}")
        print(f"    Разнообразие действий: budget={unique_budget}, compat={unique_compat}, profile={unique_profile}")
    
    print(f"\n📊 Средний reward: {sum(total_rewards)/len(total_rewards):.1f}")
