FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ECOROUTE_DB=/data/ecoroute.db
COPY pyproject.toml README.md ./
COPY ecoroute ./ecoroute
RUN pip install --no-cache-dir . && mkdir -p /data
EXPOSE 8000
VOLUME ["/data"]
CMD ["ecoroute", "serve", "--host", "0.0.0.0", "--port", "8000"]
