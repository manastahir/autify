FROM python:3.7.13-slim-buster
RUN mkdir -p /src/experiments
WORKDIR /src
COPY ./requirements.txt /src/requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get -y install zip
RUN apt-get -y install wget
RUN wget -O /src/data.zip https://storage.googleapis.com/autify-dev-dl/data/checkbox_state_v2.zip
RUN ls
COPY . /src
RUN unzip -q data.zip
CMD python train.py
