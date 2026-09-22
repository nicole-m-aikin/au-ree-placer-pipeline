# Gold-placer lookalike doorbell. Chemistry in, P(lookalike) out.
# Not a national model. Not Task 4 tonnes.
FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ api/
COPY pipeline/__init__.py pipeline/__init__.py
COPY pipeline/ml_artifacts.py pipeline/ml_artifacts.py
COPY pipeline/ml_preprocess.py pipeline/ml_preprocess.py
COPY pipeline/ml_spatial.py pipeline/ml_spatial.py
COPY models/ models/

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
