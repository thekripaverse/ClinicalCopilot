FROM python:3.11-slim

# System deps for opencv, etc.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY wheels/ ./wheels/
ENV PIP_DEFAULT_TIMEOUT=120

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --find-links=./wheels -r requirements.txt

# Copy the project
COPY . .

# Set env vars if you use any secret keys (example)
# ENV OPENAI_API_KEY=your_key_here

# Expose FastAPI port
EXPOSE 8000

# Run app
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
