# src/backend/agent_pipeline.py
"""
Оркестрация агентов для генерации корзины.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import time

# Добавляем корень проекта в PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.compatibility.agent import CompatibilityAgent
from src.agents.budget.agent import BudgetAgent
from src.nlp.llm_parser import parse_query_with_function_calling
from src.schemas.basket_item import BasketItem  


# src/backend/agent_pipeline.py

class AgentPipeline:
    """Пайплайн для последовательной обработки запроса агентами."""
    
    def __init__(self):
        """Инициализирует агентов."""
        print("   🤖 Загрузка Compatibility Agent...")
        self.compatibility_agent = CompatibilityAgent()
        
        print("   💰 Загрузка Budget Agent...")
        self.budget_agent = BudgetAgent()
        
        print("   👤 Profile Agent (заглушка)...")
        self.profile_agent = None  # TODO
    
    
    def process(self, user_query: str) -> Dict[str, Any]:
        """
        Обрабатывает запрос через весь пайплайн.
        
        Args:
            user_query: Запрос пользователя в свободной форме
            
        Returns:
            Результат обработки с корзиной и метаданными
        """
        start_time = time.time()
        stages = []
        parsed_query = {}
        
        try:
            # ============================================
            # ЭТАП 1: LLM PARSER
            # ============================================
            print("\n🧠 ЭТАП 1: LLM Parser")
            stage1_start = time.time()
            
            parsed_query = parse_query_with_function_calling(user_query)
            
            budget_rub = parsed_query.get('budget_rub') or 3000
            people = parsed_query.get('people') or 2
            meal_types = parsed_query.get('meal_type') or ['dinner']
            
            print(f"   ✅ Распознано: {parsed_query}")
            print(f"   💡 Применены дефолты: people={people}, budget={budget_rub}, meals={meal_types}")
            
            stages.append({
                'agent': 'llm_parser',
                'name': '🧠 LLM Parser',
                'status': 'completed',
                'duration': round(time.time() - stage1_start, 2),
                'result': {'parsed': parsed_query}
            })
            
            # ============================================
            # ЭТАП 2: COMPATIBILITY AGENT
            # ============================================
            print("\n🔗 ЭТАП 2: Compatibility Agent")
            stage2_start = time.time()
            
            compatibility_query = {
                'meal_types': meal_types,
                'people': people,
                'budget_rub': budget_rub,
                'exclude_tags': parsed_query.get('exclude_tags', []),
                'include_tags': parsed_query.get('include_tags', [])
            }
            
            compatibility_result = self.compatibility_agent.generate_basket(
                parsed_query=compatibility_query,
                strategy='smart'  
            )
            
            basket_v1: List[BasketItem] = compatibility_result.get('basket', [])
            
            print(f"   ✅ Найдено товаров: {len(basket_v1)}")
            print(f"   💵 Итого: {compatibility_result.get('total_price', 0):.2f}₽")
            
            stages.append({
                'agent': 'compatibility',
                'name': '🔗 Compatibility Agent',
                'status': 'completed',
                'duration': round(time.time() - stage2_start, 2),
                'result': {
                    'basket': basket_v1,
                    'scenario': compatibility_result.get('scenario_used'),
                    'compatibility_score': compatibility_result.get('compatibility_score'),
                    'total_price': compatibility_result.get('total_price'),
                    'success': compatibility_result.get('success')
                }
            })
            
            basket_current = basket_v1
            
            # ============================================
            # ЭТАП 3: BUDGET AGENT
            # ============================================
            print("\n💰 ЭТАП 3: Budget Agent")
            stage3_start = time.time()
            
            budget_result = self.budget_agent.optimize(
                basket=basket_current,  # ✅ Передаем List[BasketItem]
                budget_rub=budget_rub,
                min_discount=0.2
            )
            
            basket_v2: List[BasketItem] = budget_result['basket']
            
            print(f"   ✅ Оптимизировано товаров: {len(budget_result['replacements'])}")
            print(f"   💰 Экономия: {budget_result['saved']:.2f}₽")
            
            stages.append({
                'agent': 'budget',
                'name': '💰 Budget Agent',
                'status': 'completed',
                'duration': round(time.time() - stage3_start, 2),
                'result': {
                    'basket': basket_v2,
                    'saved': budget_result['saved'],
                    'replacements': budget_result['replacements'],
                    'within_budget': budget_result['within_budget'],
                    'optimized': len(budget_result['replacements']) > 0
                }
            })
            
            basket_current = basket_v2
            
            # ============================================
            # ЭТАП 4: PROFILE AGENT (заглушка)
            # ============================================
            print("\n👤 ЭТАП 4: Profile Agent")
            stage4_start = time.time()
            
            basket_v3 = basket_current  # ✅ Теперь basket_v3 определен!
            
            stages.append({
                'agent': 'profile',
                'name': '👤 Profile Agent',
                'status': 'completed',
                'duration': round(time.time() - stage4_start, 2),
                'result': {
                    'basket': basket_v3,
                    'personalized': False,
                    'message': 'В разработке'
                }
            })
            
            formatted_basket = []
            for item in basket_v3:
                formatted_item = {
                    **item,  # Все существующие поля
                    'price_display': f"{item['price_per_unit']:.2f}₽/{item['unit']}",
                    'quantity_display': f"{item['quantity']:.2f}{item['unit']}",
                    'total_display': f"{item['total_price']:.2f}₽",
                    'breakdown': f"{item['quantity']:.2f}{item['unit']} × {item['price_per_unit']:.2f}₽ = {item['total_price']:.2f}₽"
                }
                formatted_basket.append(formatted_item)
            
            # ============================================
            # ФИНАЛЬНЫЙ РЕЗУЛЬТАТ
            # ============================================
            total_price = sum(item['total_price'] for item in basket_v3)
            original_price = compatibility_result.get('total_price', total_price)
            savings = original_price - total_price
            
            return {
                'status': 'success',
                'parsed': parsed_query,
                'basket': formatted_basket,  # ✅ Используем форматированную версию
                'summary': {
                    'items_count': len(basket_v3),
                    'total_price': round(total_price, 2),
                    'original_price': round(original_price, 2),
                    'savings': round(savings, 2),
                    'budget_rub': budget_rub,
                    'within_budget': total_price <= budget_rub,
                    'execution_time_sec': round(time.time() - start_time, 2)
                },
                'stages': stages,
                'metadata': {
                    'people': people,
                    'meal_types': meal_types,
                    'scenario_used': compatibility_result.get('scenario_used', {}).get('name'),
                    'strategy': 'smart'
                }
            }
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            return {
                'status': 'error',
                'message': str(e),
                'type': type(e).__name__,
                'parsed': parsed_query,
                'stages': stages
            }
