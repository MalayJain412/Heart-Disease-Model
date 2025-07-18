from flask import Blueprint, redirect, url_for
from flask_login import current_user

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role == 'doctor':
            return redirect(url_for('doctor.doctor_dashboard'))
        elif current_user.role == 'dean':
            return redirect(url_for('dean.dean_dashboard'))
        elif current_user.role == 'chairman':
            return redirect(url_for('chairperson.chairman_dashboard'))
    return redirect(url_for('auth.login')) 