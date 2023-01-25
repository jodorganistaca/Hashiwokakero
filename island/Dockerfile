# syntax=docker/dockerfile:1

FROM python:3.8-buster

WORKDIR /app
COPY . .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r ./requirements.txt

EXPOSE 5000
ENTRYPOINT [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
