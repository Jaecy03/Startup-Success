# Dockerfile

FROM python:3.12-slim

WORKDIR /app

COPY . .

# Upgrade pip and install dependencies with longer timeout and retries
RUN pip install --upgrade pip && \
    pip install --default-timeout=100 --retries=5 -r requirements.txt

CMD ["python", "main.py"]
