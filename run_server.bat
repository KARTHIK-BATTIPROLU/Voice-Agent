@echo off
call venv\Scripts\activate.bat
uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level info
