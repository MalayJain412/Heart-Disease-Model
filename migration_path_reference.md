# Migration Path Reference

This file lists the current file and folder structure, indicating which files belong to the frontend and backend. Use this as a reference to update import paths, static/template references, and API connectivity after restructuring.

## Backend
- app.py
- config.py
- startup.py
- instance/
- data/
- scripts/
- tests/
- requirements.txt
- Dockerfile
- env.example
- env.production
- render.yaml
- docker-compose.yml

## Frontend
- app/static/
  - css/
  - js/
- app/templates/
  - auth/
    - change_password.html
    - login.html
    - register_patient.html
  - base.html
  - chairman/
    - dashboard.html
    - register_user.html
  - dean/
    - analytics.html
    - dashboard.html
    - register_doctor.html
  - doctor/
    - add_patient.html
    - dashboard.html
  - patient/

## Notes
- Flask currently serves templates from `app/templates/` and static files from `app/static/`.
- Any references to static files in templates use the `url_for('static', filename='...')` function.
- API endpoints are defined in `app.py` and possibly other backend files.
- After moving, update Flask's `template_folder` and `static_folder` arguments if needed.
- Update any hardcoded paths in scripts, configs, or templates referencing the old structure. 