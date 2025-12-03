#!/bin/bash
# Run the FastAPI backend
# Load environment variables from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

echo "Starting Email AI Predictor Backend..."
echo "LLM: ${LLM_BASE_URL:-http://127.0.0.1:1234/v1}"
echo "DB: ${DATABASE_URL:-sqlite:///./email_predictor.db}"

# Check if database exists and initialize if needed
if [ ! -f "email_predictor.db" ]; then
  echo "Database not found. Initializing..."
  ./venv/bin/python3 populate_db.py
  if [ $? -eq 0 ]; then
    echo "Database initialized successfully!"
  else
    echo "Error initializing database. Please run 'python populate_db.py' manually."
    exit 1
  fi
fi

echo "Starting server..."
./venv/bin/python3 -m uvicorn main:app --reload --host ${HOST:-0.0.0.0} --port ${PORT:-8000}
