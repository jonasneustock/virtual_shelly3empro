FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py /app/app.py
COPY virtual_shelly /app/virtual_shelly
#COPY vshelly /app/vshelly
ENV PYTHONPATH=/app
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

CMD ["/bin/sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${HTTP_PORT:-80}"]
