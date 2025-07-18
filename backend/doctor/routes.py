from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from models import db, Patient
import pandas as pd

# You may need to import the model loading logic if used
# from ... import model

doctor_bp = Blueprint('doctor', __name__, url_prefix='/doctor')

@doctor_bp.route('/dashboard')
@login_required
def doctor_dashboard():
    if current_user.role != 'doctor':
        flash('Access denied', 'error')
        return redirect(url_for('auth.index'))
    patients = Patient.query.filter_by(doctor_id=current_user.id).order_by(Patient.timestamp.desc()).all()
    return render_template('doctor/dashboard.html', patients=patients)

@doctor_bp.route('/add_patient', methods=['GET', 'POST'])
@login_required
def add_patient():
    if current_user.role != 'doctor':
        flash('Access denied', 'error')
        return redirect(url_for('doctor.doctor_dashboard'))
    model = getattr(current_app, 'model', None)
    if model is None:
        flash('Model not loaded. Please contact administrator.', 'error')
        return redirect(url_for('doctor.doctor_dashboard'))
    if request.method == 'POST':
        # Collect form data
        patient_data = {
            'name': request.form.get('name'),
            'gender': request.form.get('gender'),
            'age': int(request.form.get('age')),
            'chest_pain': bool(request.form.get('chest_pain')),
            'shortness_of_breath': bool(request.form.get('shortness_of_breath')),
            'fatigue': bool(request.form.get('fatigue')),
            'systolic': int(request.form.get('systolic')),
            'diastolic': int(request.form.get('diastolic')),
            'heart_rate': int(request.form.get('heart_rate')),
            'lung_sounds': bool(request.form.get('lung_sounds')),
            'cholesterol': int(request.form.get('cholesterol')),
            'ldl': int(request.form.get('ldl')),
            'hdl': int(request.form.get('hdl')),
            'diabetes': bool(request.form.get('diabetes')),
            'atrial_fibrillation': bool(request.form.get('atrial_fibrillation')),
            'rheumatic_fever': bool(request.form.get('rheumatic_fever')),
            'mitral_stenosis': bool(request.form.get('mitral_stenosis')),
            'aortic_stenosis': bool(request.form.get('aortic_stenosis')),
            'tricuspid_stenosis': bool(request.form.get('tricuspid_stenosis')),
            'pulmonary_stenosis': bool(request.form.get('pulmonary_stenosis')),
            'dilated_cardiomyopathy': bool(request.form.get('dilated_cardiomyopathy')),
            'hypertrophic_cardiomyopathy': bool(request.form.get('hypertrophic_cardiomyopathy')),
            'drug_use': bool(request.form.get('drug_use')),
            'fever': bool(request.form.get('fever')),
            'chills': bool(request.form.get('chills')),
            'alcoholism': bool(request.form.get('alcoholism')),
            'hypertension': bool(request.form.get('hypertension')),
            'fainting': bool(request.form.get('fainting')),
            'dizziness': bool(request.form.get('dizziness')),
            'smoking': bool(request.form.get('smoking')),
            'obesity': bool(request.form.get('obesity')),
            'murmur': bool(request.form.get('murmur'))
        }
        # Prepare data for prediction
        prediction_data = {
            'Age': patient_data['age'],
            'Chest pain': int(patient_data['chest_pain']),
            'Shortness of breath': int(patient_data['shortness_of_breath']),
            'Fatigue': int(patient_data['fatigue']),
            'Systolic': patient_data['systolic'],
            'Diastolic': patient_data['diastolic'],
            'Heart rate (bpm)': patient_data['heart_rate'],
            'Lung sounds': int(patient_data['lung_sounds']),
            'Cholesterol level (mg/dL)': patient_data['cholesterol'],
            'LDL level (mg/dL)': patient_data['ldl'],
            'HDL level (mg/dL)': patient_data['hdl'],
            'Diabetes': int(patient_data['diabetes']),
            'Atrial fibrillation': int(patient_data['atrial_fibrillation']),
            'Rheumatic fever': int(patient_data['rheumatic_fever']),
            'Mitral stenosis': int(patient_data['mitral_stenosis']),
            'Aortic stenosis': int(patient_data['aortic_stenosis']),
            'Tricuspid stenosis': int(patient_data['tricuspid_stenosis']),
            'Pulmonary stenosis': int(patient_data['pulmonary_stenosis']),
            'Dilated cardiomyopathy': int(patient_data['dilated_cardiomyopathy']),
            'Hypertrophic cardiomyopathy': int(patient_data['hypertrophic_cardiomyopathy']),
            'Drug use': int(patient_data['drug_use']),
            'Fever': int(patient_data['fever']),
            'Chills': int(patient_data['chills']),
            'Alcoholism': int(patient_data['alcoholism']),
            'Hypertension': int(patient_data['hypertension']),
            'Fainting': int(patient_data['fainting']),
            'Dizziness': int(patient_data['dizziness']),
            'Smoking': int(patient_data['smoking']),
            'Obesity': int(patient_data['obesity']),
            'Murmur': int(patient_data['murmur'])
        }
        try:
            prediction_df = pd.DataFrame([prediction_data])
            prediction_result = model.predict(prediction_df)[0]
            patient = Patient(
                name=patient_data['name'],
                gender=patient_data['gender'],
                age=patient_data['age'],
                chest_pain=patient_data['chest_pain'],
                shortness_of_breath=patient_data['shortness_of_breath'],
                fatigue=patient_data['fatigue'],
                systolic=patient_data['systolic'],
                diastolic=patient_data['diastolic'],
                heart_rate=patient_data['heart_rate'],
                lung_sounds=patient_data['lung_sounds'],
                cholesterol=patient_data['cholesterol'],
                ldl=patient_data['ldl'],
                hdl=patient_data['hdl'],
                diabetes=patient_data['diabetes'],
                atrial_fibrillation=patient_data['atrial_fibrillation'],
                rheumatic_fever=patient_data['rheumatic_fever'],
                mitral_stenosis=patient_data['mitral_stenosis'],
                aortic_stenosis=patient_data['aortic_stenosis'],
                tricuspid_stenosis=patient_data['tricuspid_stenosis'],
                pulmonary_stenosis=patient_data['pulmonary_stenosis'],
                dilated_cardiomyopathy=patient_data['dilated_cardiomyopathy'],
                hypertrophic_cardiomyopathy=patient_data['hypertrophic_cardiomyopathy'],
                drug_use=patient_data['drug_use'],
                fever=patient_data['fever'],
                chills=patient_data['chills'],
                alcoholism=patient_data['alcoholism'],
                hypertension=patient_data['hypertension'],
                fainting=patient_data['fainting'],
                dizziness=patient_data['dizziness'],
                smoking=patient_data['smoking'],
                obesity=patient_data['obesity'],
                murmur=patient_data['murmur'],
                prediction_result=prediction_result,
                doctor_id=current_user.id
            )
            db.session.add(patient)
            db.session.commit()
            flash(f'Patient added successfully! Prediction: {prediction_result}', 'success')
            return redirect(url_for('doctor.doctor_dashboard'))
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'error')
            return redirect(url_for('doctor.add_patient'))
    return render_template('doctor/add_patient.html') 