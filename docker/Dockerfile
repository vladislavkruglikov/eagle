FROM python:3.11-slim

WORKDIR /opt/eagle

ENV PYTHONUNBUFFERED=1

RUN pip install --upgrade pip

COPY ./setup.py ./setup.py
COPY ./eagle ./eagle

RUN pip install .
