# src/agent/masked_agent.py
"""
Обёртка для использования обученной MaskablePPO модели в продакшене.
"""
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.agent.multi_action_masked_env import create_masked_env


class MaskedBasketAgent:
    """
    Агент для выбора товаров с использованием обученной MaskablePPO.
    """
    
    def __init__(self, model_path="models/ppo_masked_v0.zip"):
        self.model = MaskablePPO.load(model_path)
        print(f"✅ Модель загружена: {model_path}")
    
    def select_products(self, products, constraints, max_steps=10):
        """
        Выбирает товары для корзины на основе constraints.
        
        Args:
            products: список товаров (словари с полями id, product_name, price_per_unit, tags)
            constraints: словарь с budget_rub, exclude_tags, include_tags, meal_type, people
            max_steps: количество шагов симуляции (по умолчанию 10)
        
        Returns:
            dict: {
                "cart_indices": [индексы выбранных товаров],
                "total_cost": сумма корзины,
                "products": [полные объекты товаров]
            }
        """
        # Создаём окружение
        env = create_masked_env(products, constraints, max_steps=max_steps)
        
        # Оборачиваем в ActionMasker
        def mask_fn(env):
            return env.action_masks()
        
        env = ActionMasker(env, mask_fn)
        
        # Запускаем симуляцию
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
        
        # Извлекаем результат
        cart_indices = env.env.cart  # env.env потому что есть обёртка ActionMasker
        total_cost = env.env.current_sum
        
        # Формируем список товаров
        selected_products = [products[idx] for idx in cart_indices]
        
        return {
            "cart_indices": cart_indices,
            "total_cost": total_cost,
            "products": selected_products,
            "cart_size": len(cart_indices),
            "budget_ratio": total_cost / constraints.get("budget_rub", 1500)
        }


# Пример использования
if __name__ == "__main__":
    from src.backend.db.queries import fetch_candidate_products
    from src.agent.utils import pad_products_to_k
    
    # Загружаем агента
    agent = MaskedBasketAgent("models/ppo_masked_v0.zip")
    
    # Пример constraints
    constraints = {
        "budget_rub": 1500,
        "exclude_tags": ["dairy"],
        "include_tags": [],
        "meal_type": ["dinner"],
        "people": 3
    }
    
    # Получаем товары
    products = fetch_candidate_products(constraints, limit=100)
    products = pad_products_to_k(products, k=100)
    
    # Выбираем товары
    result = agent.select_products(products, constraints)
    
    print("\n" + "=" * 70)
    print("🛒 РЕЗУЛЬТАТ")
    print("=" * 70)
    print(f"Товаров в корзине: {result['cart_size']}")
    print(f"Общая стоимость: {result['total_cost']:.2f}₽")
    print(f"Использование бюджета: {result['budget_ratio']*100:.1f}%")
    print(f"\nТовары:")
    
    for i, product in enumerate(result['products'][:10], 1):
        print(f"  {i}. {product['product_name'][:50]} — {product['price_per_unit']:.2f}₽")
    
    if len(result['products']) > 10:
        print(f"  ... и ещё {len(result['products']) - 10} товаров")
