FROM python:3.11-slim

WORKDIR /app

# PySCF requires BLAS headers and a Fortran toolchain at install time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libopenblas-dev \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY workers/dft_worker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY common /app/common
COPY dft /app/dft
COPY workers/__init__.py /app/workers/__init__.py
COPY workers/dft_worker /app/workers/dft_worker

ENV PYTHONPATH=/app

CMD ["celery", "-A", "workers.dft_worker.app.celery_app", "worker", "--loglevel=info", "-Q", "dft", "--concurrency=1"]
