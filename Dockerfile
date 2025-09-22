# Base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy all files into the container
COPY . .

# Install dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Environment variables for Flask (optional but good practice)
ENV FLASK_APP=flask_app/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# Run Flask server
CMD ["python", "app/app.py"]
