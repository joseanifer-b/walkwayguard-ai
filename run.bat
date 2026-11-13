@echo off
cd /d %~dp0
if not exist frontend\dist (
  cd frontend && call npm install && call npm run build && cd ..
)
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
