from app import app, db
from werkzeug.security import generate_password_hash
from models import db, User, Recipe


# データベースに接続してデータを作成する関数
def create_sample_data():
    with app.app_context():  # Flask アプリのコンテキストを有効化
        # ユーザーを作成
        user1 = User(username="test1", password=generate_password_hash("1"))
        user2 = User(username="test2", password=generate_password_hash("2"))

        # データベースに追加
        db.session.add(user1)
        db.session.add(user2)
        db.session.commit()

        # レシピを作成
        recipe1 = Recipe(
            user_id=user1.id,
            name="Spaghetti Carbonara",
            steps=[
                "Boil water and cook spaghetti.",
                "Fry bacon until crispy.",
                "Mix eggs, cheese, and pepper in a bowl.",
                "Combine spaghetti, bacon, and egg mixture.",
            ],
            likes=True,
            ingredients=[
                {"name": "Spaghetti", "quantity": "200g"},
                {"name": "Bacon", "quantity": "100g"},
                {"name": "Eggs", "quantity": "2"},
                {"name": "Parmesan Cheese", "quantity": "50g"},
                {"name": "Black Pepper", "quantity": "to taste"},
            ],
        )

        recipe2 = Recipe(
            user_id=user1.id,
            name="Caesar Salad",
            steps=[
                "Chop lettuce and place in a bowl.",
                "Add croutons, cheese, and dressing.",
                "Mix well and serve.",
            ],
            likes=False,
            ingredients=[
                {"name": "Lettuce", "quantity": "1 head"},
                {"name": "Croutons", "quantity": "50g"},
                {"name": "Parmesan Cheese", "quantity": "30g"},
                {"name": "Caesar Dressing", "quantity": "to taste"},
            ],
        )

        recipe3 = Recipe(
            user_id=user2.id,
            name="Tomato Soup",
            steps=[
                "Chop onions and garlic.",
                "Cook onions and garlic in olive oil until soft.",
                "Add canned tomatoes and chicken broth.",
                "Simmer for 20 minutes.",
                "Blend until smooth and serve.",
            ],
            likes=True,
            ingredients=[
                {"name": "Onions", "quantity": "1"},
                {"name": "Garlic", "quantity": "2 cloves"},
                {"name": "Canned Tomatoes", "quantity": "400g"},
                {"name": "Chicken Broth", "quantity": "500ml"},
                {"name": "Olive Oil", "quantity": "2 tbsp"},
            ],
        )

        # データベースに追加
        db.session.add(recipe1)
        db.session.add(recipe2)
        db.session.add(recipe3)
        db.session.commit()

        print("Sample data created successfully!")

# メイン実行部分
if __name__ == "__main__":
    create_sample_data()
