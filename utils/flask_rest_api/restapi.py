# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Run a Flask REST API exposing one or more YOLOv5s models."""
# システム関連
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # スクリプトのディレクトリを取得
import argparse
import io

# 画像認識関連
import torch
from PIL import Image

# フロントにに影響するものとか
from flask import Flask, request, session, jsonify

# cors ポリシー
from flask_cors import CORS

# db関連
from flask_sqlalchemy import SQLAlchemy 
from datetime import datetime
from db_models import db, User, Recipe

# ログイン関連
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin

# 登録関連
from werkzeug.security import generate_password_hash, check_password_hash

# app初期化
app = Flask(__name__)
CORS(app)

# # http methodの限定
# @app.before_request
# def handle_options():
#     if request.method == "OPTIONS":
#         response = app.make_default_options_response()
#         headers = response.headers

#         headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
#         headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS, DELETE"
#         headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
#         return response 


# ログイン関連
bcrypt = Bcrypt(app)

# db関連
# db初期化
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.todo'
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'db.cook_app')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.urandom(24)
# db = SQLAlchemy()
db.init_app(app)  

# ここからルーティング設定

# トップページ
@app.route('/', methods=['GET'])
def hello():
    return "Server is running on the web!"

# 新規登録ページ
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

########################## login page ############################ 
@app.route('/login', methods=['POST'])
def login():
    data = request.json  # フロントエンドからJSONデータを受け取る
    username = data.get('username')
    password = data.get('password')

    # ユーザーが存在するか確認
    user = User.query.filter_by(username=username).first()
    print(user)
    if user and check_password_hash(user.password, password):
        # 認証成功: セッションにユーザー情報を保存
        session['user_id'] = user.id
        session['username'] = user.username
        return jsonify({"message": "Login successful!"}), 200
    else:
        # 認証失敗
        return jsonify({"message": "ユーザー名またはパスワードが違います。"}), 401
####################################################################


########################## logout page ##########################
@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)  # セッションからユーザー情報を削除
    session.pop('username', None)
    return jsonify({"message": "Logged out successfully!"}), 200
####################################################################


########################## user recipe page ##########################
@app.route('/api/users/<string:username>/recipes', methods=['GET'])
def get_user_recipes(username):
    # ユーザーの存在を確認
    user = User.query.filter_by(username=username).first()
    if not user:
        abort(404, description="User not found")

    # ユーザーのレシピを取得
    recipes = Recipe.query.filter_by(user_id=user.id).all()
    recipe_list = [{"id": recipe.id, "name": recipe.name} for recipe in recipes]

    return jsonify(recipe_list)
####################################################################


########################## ここからマコちゃんのやつ  ########################## 
# restapi.py から見て models ディレクトリが2つ上の階層にある
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# print("Current sys.path:", sys.path)

from pathlib import Path
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
import numpy as np
import pathlib
from functools import partial
from utils.augmentations import letterbox  # 必要に応じて letterbox 関数を手動定義

# WindowsPath を PosixPath に変換する関数
def map_location_fix(storage, location):
    if isinstance(location, pathlib.WindowsPath):
        return pathlib.PosixPath(location)
    return location

#app = Flask(__name__)

#モデルディレクトリを指定する位置を移動
#MODEL_DIR = "/mnt/host_files"
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')) + "/best"
# MODEL_DIR = "/yolov5/best"
models = {}
unified_labels = []
model_mapping = []

# デバイスの選択
device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
# モデルを動的にロード
'''
for file in os.listdir(MODEL_DIR):
    if file.endswith(".pt"):
        model_path = Path(MODEL_DIR) / file
        model_name = os.path.splitext(file)[0]
        try:
            # DetectMultiBackend のロード
            model = DetectMultiBackend(str(model_path), device=device)
        except Exception as e:
            print(f"モデル {file} のロード中にエラー: {e}")
            continue

        models[model_name] = model
        labels = model.names
        start_index = len(unified_labels)
        unified_labels.extend(labels)
        model_mapping.append({i: start_index + i for i in range(len(labels))})
'''
for file in os.listdir(MODEL_DIR):
    if file.endswith(".pt"):
        model_path = Path(MODEL_DIR) / file
        model_name = os.path.splitext(file)[0]
        try:
            # DetectMultiBackend のロード
            model = DetectMultiBackend(str(model_path), device=device)
        except Exception as e:
            print(f"モデル {file} のロード中にエラー: {e}")
            continue

        # モデルを `models` に格納
        models[model_name] = {
            "model": model,
            "labels": model.names,  # ラベル名を格納
        }

        labels = model.names
        start_index = len(unified_labels)
        unified_labels.extend(labels)
        model_mapping.append({i: start_index + i for i in range(len(labels))})

@app.route('/predict', methods=['POST'])
def predict():
    # フロントエンドから送信された画像ファイルを取得
    file = request.files['image']  
    file_path = f"/gaei3_back/uploads/{file.filename}"  # 一時保存用のパス
    file.save(file_path)  # 画像ファイルを保存

    # 画像処理
    img = Image.open(file_path).convert('RGB')
    img = np.array(img)
    img, ratio, pad = letterbox(img, new_shape=640, auto=False, scaleup=False)
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    all_predictions = []

    for model_name, model_data in models.items():
        model = model_data["model"]
        labels = model_data["labels"]  # ラベル名を取得

        try:
            results = model(img)  # 推論
            predictions = []
            threshold = 0.5  # 信頼度の閾値

            for det in results[0]:  # 各予測結果を処理
                if det is None or len(det) == 0:
                    continue  # 予測結果がない場合スキップ
                
                for row in det:
                    if len(row) < 5:
                        print(f"警告: row の長さが不足しています (row: {row})")
                        continue

                    bbox = tuple(row[:4].tolist())  # バウンディングボックス（タプルに変換）
                    confidence = row[4].item()  # 信頼度スコア

                    if confidence >= threshold:  # 信頼度が閾値を超えている場合のみ
                        class_scores = row[5:]
                        if class_scores.numel() > 0:
                            class_id = int(torch.argmax(class_scores))
                            label = labels[class_id]

                            # 既存の物体との重複をチェック
                            existing = next(
                                (p for p in predictions if iou(p["bbox"], bbox) > 0.5),
                                None
                            )

                            if existing:
                                # 信頼度を比較して高い方を優先
                                if confidence > existing["confidence"]:
                                    existing["confidence"] = confidence
                                    existing["label"] = label
                            else:
                                # 新規エントリとして追加
                                predictions.append({
                                    "bbox": bbox,
                                    "confidence": confidence,
                                    "label": label
                                })
                        else:
                            print("警告: クラススコアが空です (row:", row, ")")

            # 物体ごとのラベルをリストに追加
            all_predictions.extend([p["label"] for p in predictions])

        except Exception as e:
            print(f"モデル {model_name} の推論中にエラー: {e}")
            all_predictions.append({"model": model_name, "error": str(e)})

    # **同じラベルを1つのみにする**
    unique_labels = list(set(all_predictions))

    print(unique_labels)
    return jsonify(unique_labels)


def iou(bbox1, bbox2):
    """2つのバウンディングボックスのIoU（交差部分の割合）を計算"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # 交差領域の幅と高さを計算
    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection_area = intersection_width * intersection_height

    # 各バウンディングボックスの面積を計算
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # IoUを計算
    union_area = bbox1_area + bbox2_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0

    

    #return jsonify(["にんじん"])
####################################################################


####################################################################
if __name__ == "__main__":
    # create allは一回やれば良さげ？
    # with app.app_context():  # app_contextを使用
    #     db.create_all()
    # parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    # parser.add_argument("--port", default=5000, type=int, help="port number")
    # parser.add_argument("--model", nargs="+", default=["yolov5s"], help="model(s) to run, i.e. --model yolov5n yolov5s")
    # opt = parser.parse_args()

    # for m in opt.model:
    #     models[m] = torch.hub.load("ultralytics/yolov5", m, force_reload=True, skip_validation=True)

    app.run(debug=True, host='0.0.0.0', port=5000)
    # app.run(host="0.0.0.0", port=opt.port)  # debug=True causes Restarting with stat
####################################################################