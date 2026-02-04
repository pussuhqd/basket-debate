# src/rl/train_compatibility.py
"""
Обучение Compatibility Agent (первый агент в Sequential pipeline).
"""
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.backend.db.queries import fetch_candidate_products
from src.agent.compatibility_env import CompatibilityEnv
import numpy as np



def mask_fn(env):
    """Функция для ActionMasker wrapper"""
    return env.action_masks()



def make_env(seed=0):
    """Создание окружения для Compatibility Agent"""
    constraints = {
        "budget_rub": 1500,
        "exclude_tags": [],  # Compatibility Agent игнорирует теги (это задача Profile Agent)
        "include_tags": [],
        "meal_type": ["dinner"],
        "people": 2,
    }
    
    # Загружаем товары С meal_components
    products = fetch_candidate_products(
        constraints, 
        limit=100,
        require_meal_components=True  # ← Только товары с meal_components
    )
    
    print(f"[INFO] Загружено {len(products)} товаров для Compatibility Agent")
    
    # Проверяем распределение meal_components
    from collections import Counter
    all_components = []
    for p in products:
        all_components.extend(p['meal_components'])
    
    print(f"[INFO] Распределение meal_components:")
    for comp, count in Counter(all_components).most_common(10):
        print(f"  {comp}: {count}")
    
    env = CompatibilityEnv(products, constraints, max_steps=15)
    env = ActionMasker(env, mask_fn)
    
    return env



if __name__ == "__main__":
    print("=" * 70)
    print("🎭 Обучение Compatibility Agent")
    print("=" * 70)
    
    env = make_env()
    
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,      # Немного ниже, чем у вас (было 5e-4)
        n_steps=2048,
        batch_size=64,           # Меньше batch для стабильности
        ent_coef=0.5,            # Exploration
        clip_range=0.2,
        gamma=0.99,
        tensorboard_log="./logs/compatibility_agent/"
    )
    
    print("\n🚀 Обучаем 100k шагов...")
    model.learn(total_timesteps=100_000, progress_bar=True)
    
    model.save("models/compatibility_agent_v0")
    print("✅ Модель сохранена: models/compatibility_agent_v0.zip")
    
    # Тестирование
    print("\n🧪 Тестируем 5 эпизодов...")
    total_rewards = []
    
    for ep in range(5):
        obs, _ = env.reset()
        ep_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        
        total_rewards.append(ep_reward)
        
        coverage = info['component_coverage']
        required = ['main_course', 'side_dish', 'beverage']
        required_met = all(coverage.get(comp, False) for comp in required)
        
        print(f"\n  Эпизод {ep+1}:")
        print(f"    reward={ep_reward:.1f}, cart={info['cart_size']}, cost={info['total_cost']:.2f}₽")
        print(f"    Required components: {'✅ ВСЕ' if required_met else '❌ НЕ ВСЕ'}")
        print(f"    Coverage: {[k for k, v in coverage.items() if v]}")
        
        # Отображение корзины
        env.unwrapped.render()
    
    print(f"\n📊 Средний reward: {np.mean(total_rewards):.1f}")
