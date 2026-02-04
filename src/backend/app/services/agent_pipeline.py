"""
Оркестрация агентов для генерации корзины (без WebSocket).
"""

from typing import Dict, Any
import time

from src.agent.compatibility.agent import CompatibilityAgent
from src.nlp.llm_parser import parse_query_with_function_calling


class AgentPipeline:
    """
    Пайплайн для последовательной обработки запроса агентами.
    """
    
    def __init__(self):
        """Инициализирует агентов."""
        print("🤖 Инициализация агентов...")
        self.compatibility_agent = CompatibilityAgent()
        self.budget_agent = None  # TODO
        self.profile_agent = None  # TODO
    
    
    def process(self, user_query: str) -> Dict[str, Any]:
        """
        Обрабатывает запрос через весь пайплайн.
        
        Args:
            user_query: Запрос пользователя
        
        Returns:
            Dict: {
                "status": "success",
                "parsed": {...},
                "basket": [...],
                "stages": [...]  # результаты каждого агента
            }
        """
        start_time = time.time()
        stages = []
        
        try:
            # ============================================
            # ЭТАП 1: LLM PARSER
            # ============================================
            stage1_start = time.time()
            
            parsed_query = parse_query_with_function_calling(user_query)
            
            # Дефолты
            if not parsed_query.get('budget_rub'):
                parsed_query['budget_rub'] = 3000
            if not parsed_query.get('people'):
                parsed_query['people'] = 1
            if not parsed_query.get('meal_type') or len(parsed_query['meal_type']) == 0:
                parsed_query['meal_type'] = ['dinner']
            
            stages.append({
                'agent': 'llm_parser',
                'name': '🧠 LLM Parser',
                'status': 'completed',
                'duration': round(time.time() - stage1_start, 2),
                'result': {
                    'parsed': parsed_query
                }
            })
            
            # ============================================
            # ЭТАП 2: COMPATIBILITY AGENT
            # ============================================
            stage2_start = time.time()
            
            compatibility_query = {
                'meal_types': parsed_query.get('meal_type', ['dinner']),
                'people': parsed_query.get('people', 1),
                'budget_rub': parsed_query.get('budget_rub'),
                'exclude_tags': parsed_query.get('exclude_tags', []),
                'include_tags': parsed_query.get('include_tags', [])
            }
            
            compatibility_result = self.compatibility_agent.generate_basket(
                parsed_query=compatibility_query,
                strategy='random'
            )
            
            basket_v1 = compatibility_result.get('basket', [])
            
            # Преобразуем формат товаров для фронта
            basket_v1_formatted = []
            for item in basket_v1:
                basket_v1_formatted.append({
                    'id': item.get('id'),
                    'name': item.get('product_name'),
                    'category': item.get('product_category', ''),
                    'brand': item.get('brand', ''),
                    'price': item.get('total_price', 0),
                    'unit': item.get('unit', ''),
                    'quantity': item.get('quantity_needed', 1),
                    'agent': 'compatibility',
                    'reason': f"Ингредиент: {item.get('ingredient_role', 'основной')}",
                    'rating': 4.5,
                    'search_score': item.get('search_score', 0)
                })
            
            stages.append({
                'agent': 'compatibility',
                'name': '🔗 Compatibility Agent',
                'status': 'completed',
                'duration': round(time.time() - stage2_start, 2),
                'result': {
                    'basket': basket_v1_formatted,
                    'scenario': compatibility_result.get('scenario_used'),
                    'compatibility_score': compatibility_result.get('compatibility_score'),
                    'total_price': compatibility_result.get('total_price'),
                    'success': compatibility_result.get('success')
                }
            })
            
            basket_current = basket_v1_formatted
            
            # ============================================
            # ЭТАП 3: BUDGET AGENT (заглушка)
            # ============================================
            stage3_start = time.time()
            
            # TODO: Реальный BudgetAgent
            basket_v2 = basket_current  # Пока без изменений
            
            stages.append({
                'agent': 'budget',
                'name': '💰 Budget Agent',
                'status': 'completed',
                'duration': round(time.time() - stage3_start, 2),
                'result': {
                    'basket': basket_v2,
                    'optimized': False,
                    'message': 'В разработке'
                }
            })
            
            basket_current = basket_v2
            
            # ============================================
            # ЭТАП 4: PROFILE AGENT (заглушка)
            # ============================================
            stage4_start = time.time()
            
            # TODO: Реальный ProfileAgent
            basket_v3 = basket_current  # Пока без изменений
            
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
            
            # ============================================
            # ФИНАЛЬНЫЙ РЕЗУЛЬТАТ
            # ============================================
            total_price = sum(item['price'] for item in basket_v3)
            
            return {
                'status': 'success',
                'parsed': parsed_query,
                'basket': basket_v3,
                'summary': {
                    'items_count': len(basket_v3),
                    'total_price': round(total_price, 2),
                    'original_price': round(total_price * 1.2, 2),
                    'savings': round(total_price * 0.2, 2),
                    'budget_rub': parsed_query.get('budget_rub'),
                    'within_budget': total_price <= parsed_query.get('budget_rub', float('inf')),
                    'execution_time_sec': round(time.time() - start_time, 2)
                },
                'stages': stages  # История работы агентов
            }
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            return {
                'status': 'error',
                'message': str(e),
                'parsed': parsed_query if 'parsed_query' in locals() else None,
                'stages': stages
            }
