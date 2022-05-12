FROM python:3.7.13-slim-buster
RUN mkdir -p /src/experiments
RUN mkdir -p /src/data
WORKDIR /src
COPY ./requirements.txt /src/requirements.txt
RUN pip install -r requirements.txt
RUN mkdir -p /src/input
RUN mkdir -p /src/output
COPY . /src
CMD python infer.py
