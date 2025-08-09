# init_db.py
from models import db
from app import app

with app.app_context():
    db.drop_all()       # Just in case, to fully clean
    db.create_all()
    print("✅ users.db created successfully.")
