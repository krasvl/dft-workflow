FROM python:3.11-slim

WORKDIR /app

# PySCF and torch-geometric's optional compiled extensions require a
# C/C++/Fortran toolchain and BLAS headers at install time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libopenblas-dev \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY workers/training_worker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY common /app/common
COPY models /app/models
COPY workers/__init__.py /app/workers/__init__.py
COPY workers/training_worker /app/workers/training_worker

ENV PYTHONPATH=/app

CMD ["celery", "-A", "workers.training_worker.app.celery_app", "worker", "--loglevel=info", "-Q", "training", "--concurrency=1"]
