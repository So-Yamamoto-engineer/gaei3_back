# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_migrate import Migrate

db = SQLAlchemy()

# Userモデルの定義
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(128), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    recipes = db.relationship('Recipe', backref='author', lazy=True)

# Recipeモデルの定義
class Recipe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(128), nullable=False)
    steps = db.Column(db.JSON, nullable=False)
    likes = db.Column(db.Boolean, default=False)
    ingredients = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# flask db init
# flask db migrate -m "Add Recipe table"
# flask db upgrade
# from app import db
# print(db.engine.table_names())  # 既存のテーブル一覧を出力
# flask db downgrade base  # 全ての変更を元に戻す
# flask db upgrade          # 再度適用x