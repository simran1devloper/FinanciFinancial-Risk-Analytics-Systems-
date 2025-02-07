FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y netcat-openbsd dnsutils

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x wait-for-it.sh

CMD ["./wait-for-it.sh", "milvus", "--", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]