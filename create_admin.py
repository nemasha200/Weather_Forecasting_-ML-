# create_admin.py
from app_config import create_app, db
from models import User
from werkzeug.security import generate_password_hash

app = create_app()

with app.app_context():
    db.create_all()

    if not User.query.filter_by(username="admin").first():
        user = User(username="admin", password=generate_password_hash("admin123"))
        db.session.add(user)
        db.session.commit()
        print("✅ Admin user created.")
    else:
        print("⚠️ Admin already exists.")
