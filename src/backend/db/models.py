import sqlite3
import os
from flask import current_app

def init_db(app):
    with app.app_context():
        db_path = current_app.config['DATABASE']
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                price REAL NOT NULL,
                category TEXT,
                embedding BLOB  -- Для SentenceTransformer (384d vec)
            )
        ''')
        # Mock data
        mock_products = [
            ('паста', 200.0, 'ужин'),
            ('соус томатный', 150.0, 'ужин'),
            ('салат', 300.0, 'ужин')
        ]
        c.executemany("INSERT OR IGNORE INTO products (name, price, category) VALUES (?, ?, ?)", mock_products)
        conn.commit()
        conn.close()
