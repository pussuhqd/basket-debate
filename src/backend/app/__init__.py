from flask import Flask, current_app, jsonify, request
import sqlite3
import os
from src.backend.db.models import init_db
from src.agent import create_basket_env
from flask_cors import CORS
import numpy as np
from src.nlp.llm_parser import parse_query_with_function_calling
from src.backend.db.queries import fetch_candidate_products  
from dotenv import load_dotenv

load_dotenv()

def to_jsonable(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def create_app():
    app = Flask(__name__, instance_relative_config=True)
    CORS(app, resources={
        r"/*": {
            "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    })
    
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(os.path.dirname(__file__), '../../data/products.db'),
        REDIS_URL='redis://localhost:6379/0'
    )
    
    init_db(app)
    
    @app.route('/')
    def index():
        return jsonify({
            "message": "MAS Basket API ready",
            "endpoints": [
                "/health",
                "/products",
                "/simulate",
                "/api/parse-and-optimize"
            ]
        })
    
    @app.route('/health')
    def health():
        return jsonify({"status": "Flask + SQLite ready"})
    
    @app.route('/products')
    def list_products():
        """Получить список товаров из БД."""
        db_path = current_app.config['DATABASE']
        with sqlite3.connect(db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT id, name, price FROM products LIMIT 10")
            products = [
                {"id": row[0], "name": row[1], "price": row[2]} 
                for row in c.fetchall()
            ]
        return jsonify(products)
    @app.route('/optimize_test', methods=['POST']) 
    def simulate_basket():
        try:
            data = request.get_json() or {}
            budget = float(data.get('budget', 1500))
            max_steps = int(data.get('max_steps', 5))
            
            env = create_basket_env(budget=budget, max_steps=max_steps)
            obs, infos = env.reset(seed=42)
            
            total_rewards = {agent: 0.0 for agent in env.possible_agents}
            
            while env.agents:
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
                obs, rewards, terms, truncs, infos = env.step(actions)
                
                for agent in rewards:
                    total_rewards[agent] += rewards[agent]
            
            env.close()
            
            agent_cycle = ["budget", "compatibility", "profile"]

            basket = []
            for i, price in enumerate(env.cart):
                agent = agent_cycle[i % len(agent_cycle)]
                basket.append({
                    "id": i + 1,
                    "name": f"Товар {i + 1}",
                    "price": float(price),
                    "agent": agent,
                    "reason": f"Выбран агентом: {agent}",
                    "rating": 4.5
                })

            total_price = float(round(env.current_sum, 2))
            items_count = int(len(env.cart))
            original_price = float(round(total_price * 1.2, 2))
            savings = float(round(original_price - total_price, 2))

            payload = {
                "status": "success",
                "basket": basket,
                "summary": {
                    "items_count": items_count,
                    "total_price": total_price,
                    "original_price": original_price,
                    "savings": savings,
                    "rewards": {k: float(round(v, 3)) for k, v in total_rewards.items()}
                }
            }

            return jsonify(to_jsonable(payload))

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/api/parse-and-optimize', methods=['POST'])
    def parse_and_optimize():
        try:
            data = request.get_json()
            user_query = data.get('query', '')
            
            if not user_query:
                return jsonify({"status": "error", "message": "Query is required"}), 400
            
            # 1. Парсим запрос через LLM
            constraints = parse_query_with_function_calling(user_query)
            print(f"[INFO] Parsed constraints: {constraints}")
            if constraints.get("budget_rub") is None:
                constraints["budget_rub"] = 3000
                print(f"[WARN] budget_rub was None, using default 1500")

            # 2. Фильтруем товары из БД
            products = fetch_candidate_products(constraints, limit=100)
            print(f"[INFO] Found {len(products)} candidate products")

            # Проверка: если товаров не найдено → ошибка
            if not products:
                return jsonify({
                    "status": "error",
                    "message": "No products found matching constraints",
                    "constraints": constraints
                }), 404

            # 3. Создаём окружение с реальными товарами
            env = create_basket_env(
                products=products,
                constraints=constraints,
                max_steps=10
            )

            # 4. Запускаем агентов
            obs, infos = env.reset(seed=42)
            total_rewards = {agent: 0.0 for agent in env.possible_agents}
            
            while env.agents:
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
                obs, rewards, terms, truncs, infos = env.step(actions)
                
                for agent in rewards:
                    total_rewards[agent] += rewards[agent]
            
            env.close()
            
            # 5. Формируем корзину с РЕАЛЬНЫМИ товарами
            agent_cycle = ["budget", "compatibility", "profile"]
            basket = []
            for i, product_idx in enumerate(env.cart):  # Теперь cart содержит ИНДЕКСЫ, а не цены
                product = products[product_idx]  # Достаём товар из списка
                agent = agent_cycle[i % len(agent_cycle)]
                basket.append({
                    "id": product["id"],  # Реальный ID из БД
                    "name": product["product_name"],  # Реальное название
                    "category": product["product_category"],  # Категория
                    "brand": product["brand"],  # Бренд
                    "price": float(product["price_per_unit"]),  # Реальная цена
                    "unit": product["unit"],  # Единица измерения (кг/л/шт)
                    "tags": product["tags"],  # Теги
                    "agent": agent,
                    "reason": f"Выбран агентом: {agent}",
                    "rating": 4.5
                })
            
            total_price = float(round(env.current_sum, 2))
            items_count = int(len(env.cart))
            original_price = float(round(total_price * 1.2, 2))
            savings = float(round(original_price - total_price, 2))
            
            # 6. Возвращаем результат
            return jsonify({
                "status": "success",
                "parsed": constraints,
                "basket": basket,
                "summary": {
                    "items_count": items_count,
                    "total_price": total_price,
                    "original_price": original_price,
                    "savings": savings,
                    "rewards": {k: float(round(v, 3)) for k, v in total_rewards.items()}
                }
            })
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                "status": "error", 
                "message": str(e)
            }), 500

    
    return app
