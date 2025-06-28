# Use Python 3.11 slim image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PORT=5000

# Set work directory
WORKDIR /app

# Install system dependencies for Azure SQL Database
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        pkg-config \
        curl \
        gnupg2 \
        apt-transport-https \
        ca-certificates \
        wget \
        software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add Microsoft repository and install ODBC Driver 18
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
    && apt-get install -y unixodbc-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p uploads
RUN mkdir -p app/static/css
RUN mkdir -p app/static/js

# Verify model file exists and show details
RUN echo "=== Model File Verification ===" && \
    ls -la heart_disease_rf_model.pkl && \
    echo "=== File size ===" && \
    du -h heart_disease_rf_model.pkl && \
    echo "=== File type ===" && \
    file heart_disease_rf_model.pkl

# Test model loading during build (if test file exists)
RUN if [ -f "test_model_loading.py" ]; then \
        echo "=== Testing Model Loading ===" && \
        python test_model_loading.py; \
    else \
        echo "=== Model Loading Test Skipped (test file not found) ===" && \
        echo "Testing basic model loading..." && \
        python -c "import pickle; model = pickle.load(open('heart_disease_rf_model.pkl', 'rb')); print('✅ Model loaded successfully!')"; \
    fi

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port (Render will override this)
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/ || exit 1

# Run startup verification and then start the application
CMD if [ -f "startup.py" ]; then \
        python startup.py && gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --access-logfile - --error-logfile - app:app; \
    else \
        echo "=== Startup verification skipped (startup.py not found) ===" && \
        gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --access-logfile - --error-logfile - app:app; \
    fi 