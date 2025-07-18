from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from models import db, User, Patient
from werkzeug.security import generate_password_hash
from collections import Counter

dean_bp = Blueprint('dean', __name__, url_prefix='/dean')

@dean_bp.route('/dashboard')
@login_required
def dean_dashboard():
    if current_user.role != 'dean':
        flash('Access denied', 'error')
        return redirect(url_for('auth.index'))
    patients = Patient.query.order_by(Patient.timestamp.desc()).all()
    doctors = User.query.filter_by(role='doctor').all()
    return render_template('dean/dashboard.html', patients=patients, doctors=doctors)

@dean_bp.route('/register_doctor', methods=['GET', 'POST'])
@login_required
def register_doctor():
    if current_user.role != 'dean':
        flash('Access denied', 'error')
        return redirect(url_for('auth.index'))
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
        elif User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
        else:
            user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password),
                role='doctor'
            )
            db.session.add(user)
            db.session.commit()
            flash('Doctor registered successfully!', 'success')
            return redirect(url_for('dean.dean_dashboard'))
    return render_template('dean/register_doctor.html')

@dean_bp.route('/analytics')
@login_required
def dean_analytics():
    if current_user.role != 'dean':
        flash('Access denied', 'error')
        return redirect(url_for('auth.index'))
    patients = Patient.query.all()
    total_patients = len(patients)
    predictions = [p.prediction_result for p in patients]
    prediction_counts = Counter(predictions)
    doctor_counts = {}
    for patient in patients:
        doctor_name = patient.doctor.username
        doctor_counts[doctor_name] = doctor_counts.get(doctor_name, 0) + 1
    return render_template('dean/analytics.html', 
                         total_patients=total_patients,
                         prediction_counts=prediction_counts,
                         doctor_counts=doctor_counts) 