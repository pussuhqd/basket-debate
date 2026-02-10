# tests/test_budget_agent.py
from agents.budget.agent import BudgetAgent

def test_calculate_total():
    agent = BudgetAgent()
    
    basket = [
        {"name": "Молоко", "price": 80.0, "quantity": 2},
        {"name": "Хлеб", "price": 40.0, "quantity": 1},
    ]
    assert agent.calculate_total(basket) == 200.0

    empty = []
    assert agent.calculate_total(empty) == 0.0

    no_qty = [{"name": "Яйца", "price": 120.0}]
    assert agent.calculate_total(no_qty) == 120.0


def test_check_budget():
    agent = BudgetAgent()
    
    basket = [
        {"name": "Молоко", "price": 80.0, "quantity": 2},  # 160
        {"name": "Хлеб", "price": 40.0, "quantity": 1},    # 40 → итого 200
    ]
    
    result_ok = agent.check_budget(basket, budget=300.0)
    assert result_ok["fits"] is True
    assert result_ok["overspend"] == 0.0
    assert result_ok["total"] == 200.0
    
    result_bad = agent.check_budget(basket, budget=150.0)
    assert result_bad["fits"] is False
    assert result_bad["overspend"] == 50.0
    assert result_bad["total"] == 200.0

def test_calculate_total_basket_item_format():
    """Тест calculate_total с реальным форматом BasketItem"""
    from agents.budget.agent import BudgetAgent
    
    agent = BudgetAgent()
    
    # Формат BasketItem с total_price
    basket_with_total = [
        {
            "id": 1,
            "name": "Молоко 3.2%",
            "price_per_unit": 85.5,
            "quantity": 2,
            "total_price": 171.0,
            "unit": "л"
        },
        {
            "id": 2,
            "name": "Хлеб белый",
            "price_per_unit": 45.0,
            "quantity": 1,
            "total_price": 45.0,
            "unit": "шт"
        }
    ]
    
    total = agent.calculate_total(basket_with_total)
    print(f"\n✅ Тест BasketItem с total_price")
    print(f"   Ожидаем: 216.0₽ (171 + 45)")
    print(f"   Получили: {total}₽")
    assert total == 216.0
    
    # Формат BasketItem БЕЗ total_price (вычисляем сами)
    basket_without_total = [
        {
            "id": 1,
            "name": "Молоко 3.2%",
            "price_per_unit": 85.5,
            "quantity": 2,
            "unit": "л"
        },
        {
            "id": 2,
            "name": "Хлеб белый",
            "price_per_unit": 45.0,
            "quantity": 1,
            "unit": "шт"
        }
    ]
    
    total2 = agent.calculate_total(basket_without_total)
    print(f"\n✅ Тест BasketItem БЕЗ total_price")
    print(f"   Ожидаем: 216.0₽ (85.5*2 + 45*1)")
    print(f"   Получили: {total2}₽")
    assert total2 == 216.0
    
    # Смешанный формат (на всякий случай)
    mixed_basket = [
        {"price": 100.0, "quantity": 1},  # старый формат
        {"price_per_unit": 50.0, "quantity": 2, "total_price": 100.0}  # новый
    ]
    
    total3 = agent.calculate_total(mixed_basket)
    print(f"\n✅ Тест смешанного формата")
    print(f"   Ожидаем: 200.0₽ (100 + 100)")
    print(f"   Получили: {total3}₽")
    assert total3 == 200.0
    
    print("\n🎉 Все тесты на BasketItem прошли!")
