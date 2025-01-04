# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Run a Flask REST API exposing one or more YOLOv5s models."""
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # スクリプトのディレクトリを取得

import argparse
import io

import torch
from flask import Flask, request, session, jsonify
from PIL import Image

from flask_cors import CORS

# db関連
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_migrate import Migrate

# ログイン関連
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin

# 登録関連
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app)
@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        response = app.make_default_options_response()
        headers = response.headers

        headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
        headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS, DELETE"
        headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return response


# ログイン関連
bcrypt = Bcrypt(app)

# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.todo'
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'db.cook_app')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.urandom(24)

db = SQLAlchemy()
db.init_app(app)  
migrate = Migrate(app, db)

class ToDo(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	todo = db.Column(db.String(128), nullable=False)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(128), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    recipes = db.relationship('Recipe', backref='author', lazy=True)  # リレーション

class Recipe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # 作った人
    name = db.Column(db.String(128), nullable=False)  # 料理名
    steps = db.Column(db.JSON, nullable=False)  # レシピの手順（配列として格納）
    likes = db.Column(db.Boolean, default=False)  # 自分用の「いいね」
    ingredients = db.Column(db.JSON, nullable=True)  # 食材リスト（オプション）
    image_url = db.Column(db.String(256), nullable=True)  # 画像のURL（オプション）
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # 作成日時
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # 更新日時
    description = db.Column(db.Text, nullable=True)  # 説明（オプション）
# flask db init
# flask db migrate -m "Add Recipe table"
# flask db upgrade
# from app import db
# print(db.engine.table_names())  # 既存のテーブル一覧を出力
# flask db downgrade base  # 全ての変更を元に戻す
# flask db upgrade          # 再度適用


@app.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    print(f"Received username: {username}, password: {password}")

    if User.query.filter_by(username=username).first():
        return jsonify({"message": "Username already exists"}), 400

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

    new_user = User(username=username, password=hashed_password)

    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User registered successfully!"}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.json  # フロントエンドからJSONデータを受け取る
    username = data.get('username')
    password = data.get('password')

    # ユーザーが存在するか確認
    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
        # 認証成功: セッションにユーザー情報を保存
        session['user_id'] = user.id
        session['username'] = user.username
        return jsonify({"message": "Login successful!"}), 200
    else:
        # 認証失敗
        return jsonify({"message": "ユーザー名またはパスワードが違います。"}), 401


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)  # セッションからユーザー情報を削除
    session.pop('username', None)
    return jsonify({"message": "Logged out successfully!"}), 200



models = {}
DETECTION_URL = "/v1/object-detection/<model>"


@app.route(DETECTION_URL, methods=["POST"])
def predict(model):
    """Predict and return object detections in JSON format given an image and model name via a Flask REST API POST
    request.
    """
    if request.method != "POST":
        return

    if request.files.get("image"):
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        if model in models:
            results = models[model](im, size=640)  # reduce size=320 for faster inference
            return results.pandas().xyxy[0].to_json(orient="records")


if __name__ == "__main__":
    with app.app_context():  # app_contextを使用
        db.create_all()
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument("--model", nargs="+", default=["yolov5s"], help="model(s) to run, i.e. --model yolov5n yolov5s")
    opt = parser.parse_args()

    for m in opt.model:
        models[m] = torch.hub.load("ultralytics/yolov5", m, force_reload=True, skip_validation=True)

    app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat
