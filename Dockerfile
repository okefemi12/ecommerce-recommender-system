FROM python:3.9-slim

WORKDIR /app

# 1. Install basics
RUN apt-get update && apt-get install -y \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy Code
COPY . .

# 4. Set Python Path
# This ensures Python understands that /app is the root of your project
ENV PYTHONPATH=/app

# --- CRITICAL FIX FOR WINDOWS USERS ---
# This removes the invisible "CR" characters that cause the "no such file" error
RUN sed -i 's/\r$//' entrypoint.sh

# 5. Permissions & Ports
RUN chmod +x entrypoint.sh
EXPOSE 8000

CMD ["./entrypoint.sh"]

