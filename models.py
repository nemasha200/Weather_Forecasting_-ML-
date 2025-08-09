from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)

    username = db.Column(db.String(120), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)

    mobile = db.Column(db.String(20), nullable=True)
    country_code = db.Column(db.String(10), nullable=True)

    role = db.Column(db.String(20), default='user', nullable=True)
    password = db.Column(db.String(255), nullable=False)  # hashed
