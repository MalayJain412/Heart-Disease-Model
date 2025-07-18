from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from werkzeug.security import generate_password_hash
from models import db, User, Patient

chairperson_bp = Blueprint('chairperson', __name__, url_prefix='/chairman')

@chairperson_bp.route('/dashboard')
@login_required
def chairman_dashboard():
    if current_user.role != 'chairman':
        flash('Access denied', 'error')
        return redirect(url_for('auth.index'))
    patients = Patient.query.order_by(Patient.timestamp.desc()).all()
    users = User.query.all()
    return render_template('chairman/dashboard.html', patients=patients, users=users)

@chairperson_bp.route('/register_user', methods=['GET', 'POST'])
@login_required
def register_user():
    if current_user.role != 'chairman':
        flash('Access denied', 'error')
        return redirect(url_for('auth.index'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
        elif User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
        else:
            user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password),
                role=role
            )
            db.session.add(user)
            db.session.commit()
            flash(f'{role.title()} registered successfully!', 'success')
            return redirect(url_for('chairperson.chairman_dashboard'))
    return render_template('chairman/register_user.html') 