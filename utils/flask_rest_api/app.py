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
from models import db, User, Recipe

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

# db関連
db = SQLAlchemy()
db.init_app(app)  

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
    # parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    # parser.add_argument("--port", default=5000, type=int, help="port number")
    # parser.add_argument("--model", nargs="+", default=["yolov5s"], help="model(s) to run, i.e. --model yolov5n yolov5s")
    # opt = parser.parse_args()

    # for m in opt.model:
    #     models[m] = torch.hub.load("ultralytics/yolov5", m, force_reload=True, skip_validation=True)

    app.run(debug=True, host='0.0.0.0', port=5000)
    # app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat
