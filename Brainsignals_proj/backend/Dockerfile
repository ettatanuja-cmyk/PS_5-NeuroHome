FROM python:3.11-slim

WORKDIR /code

# System deps for scipy / torch / sklearn
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY Home-Appliance-Control-Dataset-main/ ./Home-Appliance-Control-Dataset-main/

EXPOSE 8000

WORKDIR /code/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
