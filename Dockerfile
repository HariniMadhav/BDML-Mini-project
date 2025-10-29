# -------------------------------
# STEP 1: Use official Python image
# -------------------------------
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# -------------------------------
# STEP 2: Copy all files
# -------------------------------
COPY . /app

# -------------------------------
# STEP 3: Install dependencies
# -------------------------------
RUN pip install --no-cache-dir tensorflow==2.15.0 flask scikit-learn pandas numpy

# -------------------------------
# STEP 4: Expose Flask port
# -------------------------------
EXPOSE 5000

# -------------------------------
# STEP 5: Run the app
# -------------------------------
CMD ["python", "app.py"]