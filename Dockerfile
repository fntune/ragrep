FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --disable-pip-version-check --no-cache-dir --upgrade pip && \
    pip install --disable-pip-version-check --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config.toml .
COPY pyproject.toml .

ENV PYTHONPATH=/app/src
EXPOSE 8080
CMD ["uvicorn", "ragrep.server:app", "--host", "0.0.0.0", "--port", "8080"]
