"""
app.py - Main Flask application factory and entry point.
Handles app creation, configuration, db/login manager initialization, model loading, and blueprint registration.
All routes and models are modularized in their respective files.
"""
import os
import pickle
from flask import Flask
from flask_login import LoginManager
from models import db, User
from auth import auth_bp
from chairperson import chairperson_bp
from doctor import doctor_bp
from dean import dean_bp
from config import config
from main import main_bp
from export import export_bp

def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def create_app():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEMPLATE_DIR = os.path.join(BASE_DIR, 'frontend', 'templates')
    STATIC_DIR = os.path.join(BASE_DIR, 'frontend', 'static')

    app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
    config_name = os.environ.get('FLASK_ENV', 'development')
    app.config.from_object(config[config_name])

    db.init_app(app)
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    # Load the ML model and attach to app
    model_path = app.config.get('MODEL_PATH', os.path.join(BASE_DIR, 'backend', 'data', 'heart_disease_rf_model.pkl'))
    app.model = load_model(model_path)
    if app.model is None:
        print("WARNING: Model could not be loaded. Predictions will not work.")
    else:
        print("Model loaded successfully!")

    # Register Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(chairperson_bp)
    app.register_blueprint(doctor_bp)
    app.register_blueprint(dean_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(export_bp)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
