# Backend Reference Structure (Pre-Refactor)

## Directory Layout

```
backend/
  app.py                # Main Flask app, all routes, models, config
  config.py             # Configuration
  data/                 # Data files (CSV, model pkl)
  docs/                 # Documentation
  env.example, env.production
  instance/             # DB instance
  mount/src/            # Model file
  requirements.txt      # Python dependencies
  scripts/              # Utility scripts
  startup.py            # Startup logic
  tests/                # Test scripts
```

## Key Files

### app.py
- Contains:
  - App creation and configuration
  - All SQLAlchemy models (User, Patient)
  - All routes (auth, doctor, dean, chairman, patient, export, API)
  - Model loading logic
  - Database initialization

### config.py
- Environment-based configuration for Flask and database

### models
- Defined in `app.py`:
  - `User`: id, username, email, password_hash, role, created_at, relationships to patients
  - `Patient`: id, name, gender, age, medical fields, doctor_id, patient_user_id, timestamp

### Route Organization (in app.py)
- Auth routes: `/login`, `/logout`, `/change_password`, `/register_patient`
- Doctor routes: `/doctor/dashboard`, `/doctor/add_patient`
- Dean routes: `/dean/dashboard`, `/dean/register_doctor`, `/dean/analytics`
- Chairman routes: `/chairman/dashboard`, `/chairman/register_user`
- Export routes: `/export/excel`, `/export/pdf`
- API routes: `/api/analytics`

### Templates
- Located in `frontend/templates/` (auth, chairman, dean, doctor subfolders)

### Static Files
- Located in `frontend/static/`

### Database
- Models expect a `patient_user_id` column in `patient` (to be added to DB)
- Uses SQL Server (MSSQL) via SQLAlchemy

---

This file documents the backend structure and logic before modularizing with Blueprints and splitting by user role. 