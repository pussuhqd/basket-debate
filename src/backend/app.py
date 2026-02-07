# src/backend/app.py
"""
Flask API для генерации корзин.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Добавляем корень проекта в PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.agent_pipeline import AgentPipeline

load_dotenv()

pipeline = None


def create_app():
    """
    Application Factory для Flask.
    Создаёт и настраивает Flask-приложение.
    """
    global pipeline
    
    app = Flask(__name__)
    
    # CORS
    CORS(app, resources={
        r"/*": {
            "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    })
    
    # Секретный ключ
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    
    # Инициализируем пайплайн при старте (только один раз)
    if pipeline is None:
        print("🚀 Инициализация пайплайна...")
        pipeline = AgentPipeline()
        print("✅ Пайплайн готов")
    
    
    # ==================== ROUTES ====================
    
    @app.route('/')
    def index():
        """Главная страница."""
        return jsonify({
            "message": "🛒 Basket Debate API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "generate_basket": "/api/generate-basket (POST)"
            }
        })
    
    
    @app.route('/health')
    def health():
        """Health check."""
        return jsonify({
            "status": "ok",
            "service": "basket-debate-api",
            "pipeline_ready": pipeline is not None
        })
    
    
    @app.route('/api/generate-basket', methods=['POST'])
    def generate_basket():
        """
        Генерация корзины через агентов.
        
        POST /api/generate-basket
        Body:
        {
            "query": "ужин на троих за 2000 без молока"
        }
        """
        try:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    "status": "error",
                    "message": "Request body is required"
                }), 400
            
            user_query = data.get('query', '')
            
            if not user_query:
                return jsonify({
                    "status": "error",
                    "message": "Field 'query' is required"
                }), 400
            
            print(f"\n{'='*70}")
            print(f"📥 Новый запрос: {user_query}")
            print(f"{'='*70}")
            
            # Запускаем пайплайн
            result = pipeline.process(user_query)
            
            print(f"\n✅ Обработано за {result.get('summary', {}).get('execution_time_sec', 0)}с")
            print(f"{'='*70}\n")
            
            return jsonify(result)
        
        except Exception as e:
            import traceback
            print(f"\n❌ ОШИБКА:")
            traceback.print_exc()
            
            return jsonify({
                "status": "error",
                "message": str(e),
                "type": type(e).__name__
            }), 500
    
    
    return app


# ==================== MAIN ====================

if __name__ == '__main__':
    print(f"📂 Project root: {PROJECT_ROOT}")
    print(f"🐍 Python path: {sys.path[:3]}")
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
