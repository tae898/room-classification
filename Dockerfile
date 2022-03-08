FROM python:3.8.12
ENV DEBIAN_FRONTEND=noninteractive 

WORKDIR /app
COPY . .

RUN apt update
RUN apt install ffmpeg libsm6 libxext6 -y
RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install -r requirements.txt

CMD ["python3.8", "app.py"]