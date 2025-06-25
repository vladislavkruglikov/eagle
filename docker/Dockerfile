FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

WORKDIR /opt/eagle

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/eagle

RUN apt update
RUN apt install --yes python3-pip python3.12-venv
RUN python3 -m venv /venv

ENV PATH="/venv/bin:$PATH"
RUN pip install --upgrade pip

RUN pip install torch==2.7.1 transformers==4.52.4 datasets==3.6.0 accelerate==1.8.1 clearml==2.0.0
RUN pip install psutil
RUN pip uninstall -y ninja && pip install ninja
RUN MAX_JOBS=4 pip install flash-attn==2.8.0.post2 --no-build-isolation

COPY ./eagle ./eagle
