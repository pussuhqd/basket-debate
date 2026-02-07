# src/nlp/llm_parser.py
import json
import re
from typing import Dict, Any, Optional
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)


def build_manual_prompt(user_query: str) -> str:
    """
    Вручную собираем промпт для модели.
    """
    function_schema = {
        "name": "parse_basket_query",
        "description": "Функция для парсинга запроса пользователя о продуктовой корзине. ОБЯЗАТЕЛЬНО ВЫЗОВИ ЭТУ ФУНКЦИЮ для любого запроса о еде, покупках или корзине.",
        "parameters": {
            "type": "object",
            "properties": {
                "budget_rub": {
                    "type": "integer",
                    "description": "Бюджет в рублях. Примеры: 'за 1500', 'до 2000', 'максимум 3000'. Если не указан, не включай это поле."
                },
                "people": {
                    "type": "integer",
                    "description": "Количество человек. Примеры: 'на двоих' (2), 'для троих' (3), 'для семьи из 4' (4). Если не указано, не включай это поле."
                },
                "horizon_value": {
                    "type": "integer",
                    "description": "Числовое значение срока. Примеры: 'на день' (1), 'на неделю' (7), 'на месяц' (30). Если не указано, не включай это поле."
                },
                "horizon_unit": {
                    "type": "string",
                    "enum": ["day", "week", "month"],
                    "description": "Единица измерения срока. 'день' -> day, 'неделя' -> week, 'месяц' -> month. Если не указано, не включай это поле."
                },
                "meal_types": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["breakfast", "lunch", "dinner", "snack"]},
                    "description": "Тип приёма пищи. 'завтрак' -> breakfast, 'обед' -> lunch, 'ужин' -> dinner, 'перекус' -> snack. Может быть несколько. Если не указано, возвращай пустой массив []."
                },
                "exclude_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Продукты, которые НУЖНО ИСКЛЮЧИТЬ. 'без молока' -> ['dairy'], 'без мяса' -> ['meat'], 'без сахара' -> ['no_sugar'], 'без глютена' -> ['gluten_free']. Если не указано, возвращай пустой массив []."
                },
                "include_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Продукты, которые ОБЯЗАТЕЛЬНО должны быть. 'веган' -> ['vegan'], 'халяль' -> ['halal'], 'детское' -> ['children_goods']. Если не указано, возвращай пустой массив []."
                },
                "max_time_min": {
                    "type": "integer",
                    "description": "Максимальное время приготовления в минутах. 'за 30 минут' -> 30, 'максимум час' -> 60, 'быстрый ужин' -> 30."
                },
                "prefer_quick": {
                    "type": "boolean",
                    "description": "True, если пользователь просит быстрое блюдо: 'быстро', 'на скорую руку', 'быстрый ужин'."
                },
                "prefer_cheap": {
                    "type": "boolean",
                    "description": "True, если пользователь просит дешево/бюджетно: 'дешево', 'недорого', 'бюджетно', 'подешевле'."
                }
            }
        }
    }
    
    system_message = f"""Ты - помощник по парсингу запросов о покупке продуктов. 

ВСЕГДА вызывай функцию parse_basket_query для ЛЮБОГО запроса о еде или корзине.

ФОРМАТ ОТВЕТА (строго следуй этому):
Вызов функции: {{"name": "parse_basket_query", "arguments": {{...}}}}

Доступная функция: {json.dumps(function_schema, ensure_ascii=False)}

Запрос пользователя: {user_query}"""
    
    return system_message


def extract_function_call(generated_text: str) -> Optional[Dict[str, Any]]:
    """
    Извлекает JSON из ответа модели.
    """
    
    # Паттерн 1: "Вызов функции: {...}"
    pattern1 = r'Вызов функции:\s*(\{.*?\})(?:\s*<end_of_turn>|$)'
    match = re.search(pattern1, generated_text, re.DOTALL)
    
    if match:
        try:
            json_str = match.group(1)
            function_call = json.loads(json_str)
            if function_call.get("name") == "parse_basket_query":
                return function_call
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSONDecodeError (pattern1): {e}")
    
    # Паттерн 2: "Вызов функции {...}" (без двоеточия)
    pattern2 = r'Вызов функции\s+(\{.*?\})(?:\s*<end_of_turn>|$)'
    match = re.search(pattern2, generated_text, re.DOTALL)
    
    if match:
        try:
            json_str = match.group(1)
            function_call = json.loads(json_str)
            if function_call.get("name") == "parse_basket_query":
                return function_call
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSONDecodeError (pattern2): {e}")
    
    # Паттерн 3: "Вызов функции parse_basket_query с параметрами: {...}"
    pattern3 = r'Вызов функции\s+parse_basket_query\s+с параметрами:\s*(\{.*?\})(?:\s*<end_of_turn>|$)'
    match = re.search(pattern3, generated_text, re.DOTALL)
    
    if match:
        try:
            arguments = json.loads(match.group(1))
            function_call = {
                "name": "parse_basket_query",
                "arguments": arguments
            }
            return function_call
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSONDecodeError (pattern3): {e}")
    
    # Паттерн 4: Поиск с подсчётом скобок
    match = re.search(r'\{[^{]*?"name"\s*:\s*"parse_basket_query"', generated_text, re.DOTALL)
    
    if match:
        try:
            start_pos = match.start()
            brace_count = 0
            end_pos = start_pos
            
            for i, char in enumerate(generated_text[start_pos:], start=start_pos):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            json_str = generated_text[start_pos:end_pos]
            function_call = json.loads(json_str)
            return function_call
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[ERROR] JSONDecodeError (pattern4): {e}")
    
    print(f"[ERROR] Не найден JSON в ответе:")
    print(generated_text[:300])
    return None


def parse_query_with_function_calling(user_query: str) -> Dict[str, Any]:
    """
    Отправляет запрос пользователя в LLM и возвращает структурированный результат.
    """
    prompt = build_manual_prompt(user_query)
    
    try:
        response = client.chat.completions.create(
            model="gemma-2-9b-it-russian-function-calling",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.05,
            max_tokens=512,
            stop=["<end_of_turn>"]
        )
        
        generated_text = response.choices[0].message.content
        print(f"[DEBUG] LLM Response: {generated_text}")
        
        function_call = extract_function_call(generated_text)
        
        if not function_call or function_call.get("name") != "parse_basket_query":
            print("[WARNING] Модель не вызвала функцию корректно.")
            print(f"[DEBUG] Распознанный function_call: {function_call}")
            return _empty_result(user_query)
        
        args = function_call.get("arguments", {})
        
        result = {
            "raw_text": user_query,
            "budget_rub": args.get("budget_rub"),
            "people": args.get("people"),
            "horizon": None,
            "meal_type": args.get("meal_types", []),
            "exclude_tags": args.get("exclude_tags", []),
            "include_tags": args.get("include_tags", []),
            "max_time_min": args.get("max_time_min"),
            "prefer_quick": args.get("prefer_quick", False),
            "prefer_cheap": args.get("prefer_cheap", False),
        }
        
        if args.get("horizon_value") and args.get("horizon_unit"):
            result["horizon"] = {
                "value": args["horizon_value"],
                "unit": args["horizon_unit"]
            }
        
        return result
        
    except Exception as e:
        print(f"[ERROR] LLM Error: {e}")
        import traceback
        traceback.print_exc()
        return _empty_result(user_query)


def _empty_result(user_query: str) -> Dict[str, Any]:
    """Возвращает пустой результат при ошибке"""
    return {
        "raw_text": user_query,
        "budget_rub": None,
        "people": None,
        "horizon": None,
        "meal_type": [],
        "exclude_tags": [],
        "include_tags": [],
        "max_time_min": None,
        "prefer_quick": False,
        "prefer_cheap": False,

    }

def test_parser():
    """Тестирует парсер."""
    print("=" * 70)
    print("🧪 ТЕСТИРОВАНИЕ LLM Parser")
    print("=" * 70)
    
    test_queries = [
        "Ужин на двоих за 1500 рублей",
        "Веганский завтрак на троих",
        "Обед без молочки и рыбы",
        "Быстрый перекус на 500 рублей"
    ]
    
    for query in test_queries:
        print(f"\n📝 Запрос: {query}")
        result = parse_query_with_function_calling(query)
        print(f"   Результат: {json.dumps(result, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    test_parser()