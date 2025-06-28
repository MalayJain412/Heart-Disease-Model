# Heart Disease Prediction System - User Documentation

## Table of Contents
1. [Getting Started](#getting-started)
2. [User Roles and Permissions](#user-roles-and-permissions)
3. [Using the Application](#using-the-application)
4. [Features Guide](#features-guide)
5. [Troubleshooting](#troubleshooting)
6. [FAQ](#faq)

---

## Getting Started

### Accessing the Application

#### Local Development
- **URL**: `http://localhost:5000`
- **Use Case**: Development and testing

#### Production Environment
- **URL**: Your deployed application URL
- **Use Case**: Live medical practice

### First-Time Setup

#### Default Login Credentials
- **Username**: `admin`
- **Password**: `admin123`
- **Role**: Chairman (Full system access)

⚠️ **Important**: Change the default password immediately after first login for security.

#### Changing Default Password
1. Login with default credentials
2. Click your username in the top-right corner
3. Select "Change Password"
4. Enter current password: `admin123`
5. Enter new password twice
6. Click "Change Password"

---

## User Roles and Permissions

### 👨‍⚕️ Doctor Role

#### What Doctors Can Do
- ✅ Add new patients and make heart disease predictions
- ✅ View only their own patients
- ✅ Change personal password
- ✅ Access comprehensive patient input form
- ✅ View prediction results and patient history

#### What Doctors Cannot Do
- ❌ View other doctors' patients
- ❌ Access analytics dashboard
- ❌ Export patient data
- ❌ Register new users
- ❌ View system-wide statistics

#### Doctor Workflow
1. **Login** with doctor credentials
2. **Dashboard** shows your patients and quick stats
3. **Add Patient** to input new patient data
4. **Fill Form** with comprehensive clinical parameters
5. **Get Prediction** from ML model
6. **View History** of your patients

### 👨‍🏫 Dean Role

#### What Deans Can Do
- ✅ View all patients across all doctors
- ✅ Register new doctors
- ✅ Access analytics dashboard with charts
- ✅ Export patient data (Excel/PDF)
- ✅ Filter patients by date, doctor, or disease type
- ✅ Monitor doctor performance
- ✅ Change personal password

#### What Deans Cannot Do
- ❌ Register other deans or chairmen
- ❌ Delete users or patients
- ❌ Modify system settings

#### Dean Workflow
1. **Login** with dean credentials
2. **Dashboard** shows system overview
3. **Analytics** for detailed insights
4. **Register Doctors** as needed
5. **Export Data** for reporting
6. **Monitor Performance** of doctors

### 👨‍💼 Chairman Role

#### What Chairmen Can Do
- ✅ Full system access
- ✅ Register doctors and deans
- ✅ View all users and patients
- ✅ Access analytics and exports
- ✅ User management capabilities
- ✅ System administration
- ✅ Change personal password

#### What Chairmen Cannot Do
- ❌ Delete the system
- ❌ Access other systems

#### Chairman Workflow
1. **Login** with chairman credentials
2. **Dashboard** shows complete system overview
3. **User Management** to add/remove users
4. **Analytics** for strategic insights
5. **System Monitoring** for performance
6. **Administration** tasks

---

## Using the Application

### Navigation

#### Top Navigation Bar
- **Logo**: Click to return to dashboard
- **Username Dropdown**: Access profile options
  - Change Password
  - Logout

#### Sidebar Navigation (Role-specific)
- **Dashboard**: Main overview page
- **Add Patient**: Input new patient data (Doctors only)
- **Analytics**: Charts and statistics (Deans/Chairmen)
- **Register User**: Add new users (Deans/Chairmen)
- **Export**: Download data (Deans/Chairmen)

### Adding a Patient (Doctors)

#### Step 1: Access Patient Form
1. Login as a doctor
2. Click "Add Patient" from dashboard
3. You'll see a comprehensive form

#### Step 2: Patient Demographics
- **Name**: Patient's full name
- **Gender**: Male/Female
- **Age**: Patient's age in years

#### Step 3: Vital Signs
- **Systolic Pressure**: Top number of blood pressure
- **Diastolic Pressure**: Bottom number of blood pressure
- **Heart Rate**: Beats per minute

#### Step 4: Symptoms
- **Chest Pain**: Check if patient experiences chest pain
- **Shortness of Breath**: Check if patient has breathing difficulties
- **Fatigue**: Check if patient feels tired
- **Lung Sounds**: Check for abnormal lung sounds

#### Step 5: Cholesterol Levels
- **Total Cholesterol**: Overall cholesterol level
- **LDL Level**: "Bad" cholesterol
- **HDL Level**: "Good" cholesterol

#### Step 6: Medical Conditions
Check boxes for any of these conditions:
- Diabetes
- Atrial Fibrillation
- Rheumatic Fever
- Mitral Stenosis
- Aortic Stenosis
- Tricuspid Stenosis
- Pulmonary Stenosis
- Dilated Cardiomyopathy
- Hypertrophic Cardiomyopathy

#### Step 7: Lifestyle Factors
Check boxes for:
- Drug Use
- Fever
- Chills
- Alcoholism
- Hypertension
- Fainting
- Dizziness
- Smoking
- Obesity
- Murmur

#### Step 8: Submit and Get Prediction
1. Click "Submit" button
2. Wait for ML model to process
3. View prediction result
4. Patient is automatically saved

### Understanding Predictions

#### Prediction Results
- **Normal**: Patient shows no signs of heart disease
- **Disease**: Patient shows signs of heart disease

#### What the Prediction Means
- **Not a Diagnosis**: This is a screening tool, not a medical diagnosis
- **Consult Doctor**: Always consult healthcare professionals
- **Risk Assessment**: Helps identify patients needing further evaluation

### Viewing Patient History

#### Doctor View
- See only your patients
- Sort by date, name, or prediction
- View detailed patient information
- Track patient outcomes

#### Dean/Chairman View
- See all patients across all doctors
- Filter by doctor, date, or disease type
- Export patient data
- Monitor trends

### Analytics Dashboard (Deans/Chairmen)

#### Overview Statistics
- **Total Patients**: Number of patients in system
- **Normal Cases**: Patients predicted as normal
- **Disease Cases**: Patients predicted with disease
- **Percentage Distribution**: Visual breakdown

#### Doctor Performance
- **Patients per Doctor**: Bar chart showing workload
- **Success Rates**: Prediction distribution by doctor
- **Activity Trends**: Patient volume over time

#### Interactive Charts
- **Pie Charts**: Disease vs normal distribution
- **Bar Charts**: Doctor performance metrics
- **Real-time Updates**: Live data refresh

### Exporting Data

#### Excel Export
1. Navigate to Export section
2. Choose "Excel Export"
3. Apply filters if needed:
   - Date range
   - Doctor selection
   - Disease type
4. Click "Export to Excel"
5. Download .xlsx file

#### PDF Export
1. Navigate to Export section
2. Choose "PDF Export"
3. Apply filters if needed
4. Click "Export to PDF"
5. Download .pdf file

#### Export Contents
- Patient demographics
- All clinical parameters
- Prediction results
- Doctor information
- Timestamps

### User Management (Deans/Chairmen)

#### Registering New Users
1. Navigate to "Register User"
2. Fill in user details:
   - Username (unique)
   - Email (unique)
   - Password
   - Role (Doctor/Dean/Chairman)
3. Click "Register User"

#### User Roles Explained
- **Doctor**: Can add patients and view their own patients
- **Dean**: Can view all patients, register doctors, access analytics
- **Chairman**: Full system access, can register any user type

---

## Features Guide

### Dashboard Features

#### Doctor Dashboard
- **Quick Stats**: Your patient count and recent activity
- **Recent Patients**: Latest patients you've added
- **Add Patient Button**: Quick access to patient form
- **Search/Filter**: Find specific patients

#### Dean Dashboard
- **System Overview**: Total patients, doctors, predictions
- **Recent Activity**: Latest system activity
- **Quick Actions**: Register doctor, view analytics
- **Performance Metrics**: Doctor activity summary

#### Chairman Dashboard
- **Complete Overview**: All system statistics
- **User Management**: Add/remove users
- **System Health**: Performance indicators
- **Administrative Tools**: System configuration

### Form Features

#### Patient Input Form
- **Responsive Design**: Works on desktop and mobile
- **Real-time Validation**: Checks input as you type
- **Auto-save**: Prevents data loss
- **Comprehensive Fields**: 30+ clinical parameters

#### Form Validation
- **Required Fields**: Must be filled
- **Range Validation**: Age, blood pressure limits
- **Format Checking**: Email, number formats
- **Error Messages**: Clear feedback

### Security Features

#### Password Management
- **Strong Passwords**: Minimum requirements
- **Secure Hashing**: Passwords are encrypted
- **Session Management**: Automatic logout
- **Change Password**: Regular updates recommended

#### Data Protection
- **Encrypted Storage**: Database encryption
- **Access Control**: Role-based permissions
- **Audit Trail**: Track user actions
- **Secure Transmission**: HTTPS in production

---

## Troubleshooting

### Login Issues

#### "Invalid Username or Password"
**Possible Causes**:
- Typo in username or password
- Caps Lock enabled
- Account doesn't exist

**Solutions**:
1. Check spelling and case
2. Try resetting password
3. Contact administrator

#### "Account Locked"
**Possible Causes**:
- Too many failed login attempts
- Account disabled

**Solutions**:
1. Wait 15 minutes
2. Contact administrator
3. Reset password

### Form Submission Issues

#### "Form Validation Error"
**Possible Causes**:
- Required fields empty
- Invalid data format
- Out of range values

**Solutions**:
1. Check all required fields
2. Verify data format
3. Ensure values are within ranges

#### "Prediction Failed"
**Possible Causes**:
- ML model not loaded
- Server error
- Invalid input data

**Solutions**:
1. Refresh page
2. Check input data
3. Contact administrator

### Performance Issues

#### "Page Loading Slowly"
**Possible Causes**:
- Large dataset
- Network issues
- Server load

**Solutions**:
1. Wait for page to load
2. Check internet connection
3. Try again later

#### "Export Taking Too Long"
**Possible Causes**:
- Large amount of data
- Server processing time
- Network bandwidth

**Solutions**:
1. Apply filters to reduce data
2. Wait for processing
3. Try smaller date ranges

### Data Issues

#### "Patient Not Found"
**Possible Causes**:
- Patient was deleted
- Search criteria too specific
- Database error

**Solutions**:
1. Broaden search criteria
2. Check patient list
3. Contact administrator

#### "Export Data Missing"
**Possible Causes**:
- No data in date range
- Filter too restrictive
- Export error

**Solutions**:
1. Check date range
2. Remove filters
3. Try different export format

---

## FAQ

### General Questions

#### Q: Is this a medical diagnosis tool?
**A**: No, this is a screening tool that helps identify patients who may need further medical evaluation. Always consult healthcare professionals for actual diagnosis.

#### Q: How accurate is the prediction model?
**A**: The model is trained on clinical data but should be used as a screening tool, not a definitive diagnosis. Accuracy depends on data quality and patient conditions.

#### Q: Can I use this for research?
**A**: Yes, the export features allow you to download data for research purposes, following appropriate data protection guidelines.

### Technical Questions

#### Q: How do I reset my password?
**A**: Use the "Change Password" option in your user menu, or contact your administrator.

#### Q: Can I access the system from mobile devices?
**A**: Yes, the application is responsive and works on smartphones and tablets.

#### Q: How often is data backed up?
**A**: Database backups are handled by the system administrator. Contact them for backup schedules.

### User Management

#### Q: How do I add a new doctor?
**A**: Deans and Chairmen can register new doctors through the "Register User" function.

#### Q: Can I change a user's role?
**A**: Only Chairmen can modify user roles. Contact your chairman for role changes.

#### Q: What happens if I forget my password?
**A**: Contact your administrator to reset your password.

### Data and Privacy

#### Q: Is patient data secure?
**A**: Yes, the system uses encryption and secure protocols to protect patient data.

#### Q: Can I export my own patients only?
**A**: Doctors can only view their own patients. Deans and Chairmen can export all patient data.

#### Q: How long is data retained?
**A**: Data retention policies are set by your organization. Contact your administrator for details.

### Support

#### Q: Who do I contact for technical support?
**A**: Contact your system administrator or the IT department.

#### Q: How do I report a bug?
**A**: Report issues to your administrator with details about the problem and steps to reproduce it.

#### Q: Can I suggest new features?
**A**: Yes, submit feature requests to your administrator for consideration.

---

## Contact Information

For technical support or questions:
- **System Administrator**: Contact your organization's IT department
- **Emergency Issues**: Use your organization's emergency contact procedures
- **Training**: Request training sessions through your administrator

---

**Note**: This documentation is specific to your organization's implementation. For the most current information, always refer to your system administrator or IT department. 