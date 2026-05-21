FROM python:3.11-slim

WORKDIR /app

# PySCF requires BLAS headers and a Fortran toolchain at install time.
# torch-geometric's optional compiled extensions need a C/C++ toolchain to
# build wheels even though we don't load them at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        libopenblas-dev \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

COPY services/density_service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY common /app/common
COPY dft /app/dft
COPY models /app/models
COPY services/__init__.py /app/services/__init__.py
COPY services/density_service /app/services/density_service

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "services.density_service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
