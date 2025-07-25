FROM python:3.8-slim-buster

RUN apt update -y && apt install awscli -y

COPY . /app
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]