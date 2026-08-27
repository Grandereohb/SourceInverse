FROM python:3.11-slim

ARG TORCH_VERSION=2.7.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    MPLBACKEND=Agg \
    PINN_DEVICE=cpu \
    PINN_AUTO_CLOSE_PLOTS=1 \
    SOURCE_INVERSION_WORK_ROOT=/data/jobs \
    SOURCE_INVERSION_QUEUE_SIZE=8 \
    SOURCE_INVERSION_MAKE_PLOTS=0 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY deployment/requirements-api.txt /tmp/requirements-api.txt
RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements-api.txt

COPY pinn_source /app/pinn_source
COPY deployment /app/deployment

RUN mkdir -p /data/jobs \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3)" || exit 1

CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
