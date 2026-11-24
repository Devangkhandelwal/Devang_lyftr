# ----------------------------- build stage -----------------------------
# Dependencies are installed into a self contained virtualenv so the
# runtime image never needs a compiler or the pip cache.
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Build tools are needed only here; they stay out of the runtime image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------------------------- runtime stage ----------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000

# Copy the ready made virtualenv from the build stage.
COPY --from=builder /opt/venv /opt/venv

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY --chown=appuser:appuser app ./app
# Tests travel with the image so "make test" can run them in the container.
COPY --chown=appuser:appuser tests ./tests
COPY --chown=appuser:appuser pytest.ini ./pytest.ini

# The SQLite database and the disk cache live here. Creating the directory
# with the right owner means the named volume inherits that ownership the
# first time it is mounted.
RUN mkdir -p /data && chown -R appuser:appuser /data /app

USER appuser

EXPOSE 8000

# Shell form so ${PORT} is expanded at container start.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
