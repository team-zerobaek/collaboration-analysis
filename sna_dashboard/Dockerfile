FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# Install required packages
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Set the working directory
WORKDIR /app

# Command to run the application
CMD ["uvicorn", "dashboard.app:fastapi_app", "--host", "0.0.0.0", "--port", "8080"]
