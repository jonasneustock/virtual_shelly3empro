FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py
VOLUME ["/data"]

# HTTP + WS range + UDP RPC + mDNS
EXPOSE 80
EXPOSE 6010-6022
EXPOSE 1010/udp
EXPOSE 2220/udp
EXPOSE 5353/udp

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
