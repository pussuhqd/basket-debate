# src/rl/inspect_masks.py
"""
Проверка, как работают маски для каждой роли.
"""
from src.backend.db.queries import fetch_candidate_products
from src.agent.multi_action_masked_env import create_masked_env
from src.agent.utils import pad_products_to_k

K = 100

constraints = {
    "budget_rub": 1500,
    "exclude_tags": ["dairy"],
    "include_tags": [],
    "meal_type": ["dinner"],
    "people": 3,
}

products = fetch_candidate_products(constraints, limit=K)
products = pad_products_to_k(products, k=K)

env = create_masked_env(products, constraints, max_steps=10)
obs, _ = env.reset()

# Получаем маски
masks = env.action_masks()
budget_mask, compat_mask, profile_mask = masks

print("=" * 70)
print("🎭 ИНСПЕКЦИЯ ACTION MASKING")
print("=" * 70)

# Анализ budget_mask
budget_allowed = [i for i, allowed in enumerate(budget_mask) if allowed and i < len(products)]
print(f"\n1️⃣  BUDGET AGENT (дешёвые товары):")
print(f"   Доступно товаров: {len(budget_allowed)}")
print(f"   Примеры:")
for idx in budget_allowed[:5]:
    p = products[idx]
    print(f"      {idx}. {p['product_name'][:40]} — {p['price_per_unit']:.2f}₽")

# Анализ compat_mask
compat_allowed = [i for i, allowed in enumerate(compat_mask) if allowed and i < len(products)]
print(f"\n2️⃣  COMPAT AGENT (разнообразие категорий):")
print(f"   Доступно товаров: {len(compat_allowed)}")
print(f"   Примеры категорий:")
categories = set(products[i]["product_category"] for i in compat_allowed[:10])
for cat in list(categories)[:5]:
    print(f"      • {cat}")

# Анализ profile_mask
profile_allowed = [i for i, allowed in enumerate(profile_mask) if allowed and i < len(products)]
print(f"\n3️⃣  PROFILE AGENT (без dairy):")
print(f"   Доступно товаров: {len(profile_allowed)}")
print(f"   Примеры:")
for idx in profile_allowed[:5]:
    p = products[idx]
    print(f"      {idx}. {p['product_name'][:40]} — теги: {', '.join(p['tags'][:3])}")

# Проверка: есть ли товары с dairy в profile_allowed?
dairy_count = sum(
    1 for idx in profile_allowed
    if 'dairy' in products[idx]['tags']
)
print(f"\n✅ Товаров с 'dairy' в profile_mask: {dairy_count} (должно быть 0!)")
