#!/bin/bash
set -e

echo "Starting backend API..."
uvicorn backend.api:app --host 0.0.0.0 --port 8000 &

echo "Waiting for backend to load models..."
sleep 5

echo "Starting frontend..."
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8501}