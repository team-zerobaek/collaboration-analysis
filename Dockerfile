FROM python:3.9

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Set the environment variable for Hugging Face token
ENV HUGGINGFACE_TOKEN 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

# Specify the entry point
ENTRYPOINT ["python", "text_extractor.py"]
