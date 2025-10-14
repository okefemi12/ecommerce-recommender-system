# ---------- Base image ----------
FROM python:3.10-slim

# ---------- Working directory ----------
WORKDIR /app

# ---------- Copy project files ----------
COPY . .

# ---------- Install dependencies ----------
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# ---------- Expose port ----------
EXPOSE 10000

# ---------- Environment variables ----------
ENV FLASK_APP=flask_app/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# ---------- Run Gunicorn ----------
CMD gunicorn --bind 0.0.0.0:${PORT:-10000} flask_app.app:app

