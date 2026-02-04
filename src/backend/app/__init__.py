from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from dotenv import load_dotenv

from src.backend.db.models import init_db
from src.backend.app.services.agent_pipeline import AgentPipeline

load_dotenv()

# Глобальный пайплайн
pipeline = None


def create_app():
    global pipeline
    
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
    )
    
    init_db(app)
    
    # Инициализируем пайплайн
    pipeline = AgentPipeline()
    
    
    # ==================== ROUTES ====================
    
    @app.route('/')
    def index():
        return jsonify({
            "message": "MAS Basket API ready",
            "endpoints": [
                "/health",
                "/api/generate-basket"
            ]
        })
    
    @app.route('/health')
    def health():
        return jsonify({"status": "Flask ready"})
    
    
    @app.route('/api/generate-basket', methods=['POST'])
    def generate_basket():
        """
        Генерация корзины через агентов.
        
        POST /api/generate-basket
        {
            "query": "ужин на троих за 2000 без молока"
        }
        """
        try:
            data = request.get_json()
            user_query = data.get('query', '')
            
            if not user_query:
                return jsonify({
                    "status": "error",
                    "message": "Query is required"
                }), 400
            
            # Запускаем пайплайн
            result = pipeline.process(user_query)
            
            return jsonify(result)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500
    
    
    return app
