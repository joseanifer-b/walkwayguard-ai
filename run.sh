#!/usr/bin/env bash
# Serves the dashboard + API at http://localhost:8000 (builds the dashboard first if needed).
set -e
cd "$(dirname "$0")"
if [ ! -d frontend/dist ] && command -v npm >/dev/null; then
  (cd frontend && npm install && npm run build)
fi
cd backend
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
