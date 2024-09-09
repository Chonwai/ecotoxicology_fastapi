FROM python:3.9

WORKDIR /app

COPY . /app

RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    pip install -r requirements.txt