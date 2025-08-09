# forms.py
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, SelectField
from wtforms.validators import DataRequired, Email, EqualTo, Length, Regexp

PHONE_RE = r'^[0-9+\-\s]{7,20}$'

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=120)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=3, max=120)])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=120)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    mobile = StringField('Mobile Number', validators=[DataRequired(), Regexp(PHONE_RE, message='Enter a valid phone')])
    country_code = SelectField('Country Code',
        choices=[('+94', '+94'), ('+81', '+81'), ('+91', '+91'), ('+44', '+44')],
        validators=[DataRequired()])
    role = SelectField('Role',
        choices=[('user', 'User'), ('admin', 'Admin')],
        validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password',
        validators=[DataRequired(), EqualTo('password', message='Passwords must match')])
    submit = SubmitField('Register')
