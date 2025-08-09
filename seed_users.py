# seed_users.py
from app import app, db
from models import User
from werkzeug.security import generate_password_hash

with app.app_context():
    db.create_all()  # safe if already created

    # Admin
    if not User.query.filter_by(username="admin").first():
        db.session.add(User(
            username="admin",
            email="admin@example.com",
            mobile="0000000",
            country_code="+94",
            role="admin",
            password=generate_password_hash("admin123")
        ))

    # Normal user
    if not User.query.filter_by(username="kamani").first():
        db.session.add(User(
            username="kamani",
            email="kamani@example.com",
            mobile="0710000000",
            country_code="+94",
            role="user",
            password=generate_password_hash("test1234")
        ))

    db.session.commit()
    print("✅ Seeded users (admin, kamani).")
