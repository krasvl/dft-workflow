FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY services/data_service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY common /app/common
COPY services/__init__.py /app/services/__init__.py
COPY services/data_service /app/services/data_service
COPY workers/__init__.py /app/workers/__init__.py
COPY workers/dft_worker /app/workers/dft_worker

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "services.data_service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
