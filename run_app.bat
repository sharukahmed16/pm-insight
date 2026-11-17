@echo off
REM Start the PM-Insight web app at http://127.0.0.1:8000 (close this window to stop it).
cd /d "%~dp0"
if not exist logs mkdir logs
call .venv\Scripts\activate.bat
start "" http://127.0.0.1:8000
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
