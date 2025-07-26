FROM python:3.8-slim-bookworm

RUN apt update -y && apt install -y awscli

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
