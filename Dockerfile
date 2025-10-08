FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs

ENV PYTHONPATH=/app/lib

EXPOSE 8000

CMD ["uvicorn", "lib.main:app", "--host", "0.0.0.0", "--port", "8000"]