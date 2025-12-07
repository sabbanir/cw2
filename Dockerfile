
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/app

COPY requirements.txt .
RUN python -c "import sys, platform; print('PY:', sys.version); print('PLAT:', platform.platform())"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    PIP_DEFAULT_TIMEOUT=120 \
    pip install -vvv --no-cache-dir -r requirements.txt

# RUN pip install --no-cache-dir -r requirements.txt

# optional: add OS tools, or copy helper scripts if you need them
# COPY src ./src

# The job will mount code at runtime, so we don't set CMD here
