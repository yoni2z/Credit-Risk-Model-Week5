FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY data/processed/preprocessor.pkl ./data/processed/preprocessor.pkl
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]