from flask import Flask, current_app, jsonify
import sqlite3
import os
from src.backend.db.models import init_db
from src.backend.agents.env import create_basket_env
from flask_cors import CORS

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
                "/simulate"
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
    
    @app.route('/simulate')
    def simulate_basket():
        """Симуляция MAS: 3 агента оптимизируют корзину за 5 шагов."""
        try:
            env = create_basket_env(budget=1500.0, max_steps=5)
            obs, infos = env.reset(seed=42)
            
            total_rewards = {agent: 0.0 for agent in env.possible_agents}
            episode_log = []
            
            while env.agents:
                actions = {
                    agent: env.action_space(agent).sample() 
                    for agent in env.agents
                }
                obs, rewards, terms, truncs, infos = env.step(actions)
                
                for agent in rewards:
                    total_rewards[agent] += rewards[agent]
                
                episode_log.append({
                    "step": env.steps,
                    "cart_sum": round(env.current_sum, 2),
                    "rewards": {k: round(float(v), 3) for k, v in rewards.items()}
                })
            
            env.close()
            
            return jsonify({
                "status": "success",
                "final_sum": round(env.current_sum, 2),
                "cart_size": len(env.cart),
                "total_rewards": {k: round(v, 3) for k, v in total_rewards.items()},
                "steps": len(episode_log)
            })
        
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e),
                "error_type": type(e).__name__
            }), 500
    
    return app
