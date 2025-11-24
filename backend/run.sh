#!/bin/bash
# Run the FastAPI backend
# Load environment variables from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

echo "Starting Email AI Predictor Backend..."
echo "LLM: ${LLM_BASE_URL}"
echo "DB: ${DATABASE_URL}"

./venv/bin/python3 -m uvicorn main:app --reload --host ${HOST:-0.0.0.0} --port ${PORT:-8000}
