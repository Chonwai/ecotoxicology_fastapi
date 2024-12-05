FROM python:3.9

WORKDIR /app

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8888"]