FROM python:3.11-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build dependencies only if needed (kept minimal)
COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install -U pip \
    && /opt/venv/bin/pip install -r /tmp/requirements.txt

FROM python:3.11-slim AS runtime

# OCI labels
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION
LABEL org.opencontainers.image.title="Virtual Shelly Pro 3EM" \
      org.opencontainers.image.description="Home Assistant–backed Shelly Pro 3EM emulator (HTTP/WS/UDP/mDNS/Modbus)." \
      org.opencontainers.image.version=$VERSION \
      org.opencontainers.image.revision=$VCS_REF \
      org.opencontainers.image.created=$BUILD_DATE

# Optional timezone support
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

# Bring in venv + app
COPY --from=build /opt/venv /opt/venv
COPY . /app

# Create non-root user and own /data
RUN useradd -r -u 10001 -s /usr/sbin/nologin appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /data /app

VOLUME ["/data"]

# HTTP + WS range + UDP RPC + mDNS
EXPOSE 80
EXPOSE 6010-6022
EXPOSE 1010/udp
EXPOSE 2220/udp
EXPOSE 5353/udp

# Use HTTP_PORT if provided, default to 80
ENV HTTP_PORT=80
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD ["python","-c","import os,sys,urllib.request; url='http://127.0.0.1:%s/healthz'%os.environ.get('HTTP_PORT','80'); sys.exit(0 if urllib.request.urlopen(url, timeout=3).status==200 else 1)"]

USER appuser
CMD ["/bin/sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${HTTP_PORT:-80}"]
