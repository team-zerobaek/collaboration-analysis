# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install libgomp1
RUN apt-get update && apt-get install -y libgomp1

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Define environment variable
ENV NAME World

# Expose port 8080 to the outside world
EXPOSE 8080

# Run app.py when the container launches
CMD ["uvicorn", "main:main_app", "--host", "0.0.0.0", "--port", "8080"]
