@echo off
echo Starting Shim Flow Simulator...

cd /d %~dp0

if not exist .venv (
echo Creating virtual environment...
python -m venv .venv
)

call .venv\Scripts\activate

pip install -r requirements.txt

echo Launching server...
uvicorn app.main:app --port 8000 --reload

pause
