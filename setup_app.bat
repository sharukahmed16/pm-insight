@echo off
REM One-time setup of the PM-Insight web app. Double-click, or run from a terminal.
REM Safe to run again. Everything is also written to logs\setup.log.
cd /d "%~dp0"
if not exist logs mkdir logs
call .venv\Scripts\activate.bat
if errorlevel 1 ( echo Could not activate .venv & pause & exit /b 1 )

echo === 1/5 Install the database and session libraries ===
pip install sqlalchemy "psycopg[binary]" itsdangerous > logs\setup.log 2>&1
if errorlevel 1 ( type logs\setup.log & echo pip install failed & pause & exit /b 1 )
pip freeze > requirements.txt

echo === 2/5 Settings file .env ===
if exist .env goto have_env
python -c "import secrets; t=open('.env.example').read(); open('.env','w').write(t.replace('SECRET_KEY=', 'SECRET_KEY=' + secrets.token_hex(32)))"
echo.
echo  Notepad opens .env now. Type your PostgreSQL password after PGPASSWORD=
echo  (the one you chose when installing PostgreSQL), save, and close Notepad.
echo.
start /wait notepad .env
:have_env

echo === 3/5 Create the database and tables ===
:init_db
python -m app.init_db
if not errorlevel 1 goto db_ok
echo.
echo  Notepad opens .env again. Correct PGPASSWORD, save, and close Notepad to retry.
echo  (Close this window instead to stop.)
start /wait notepad .env
goto init_db
:db_ok

echo === 4/5 Demo data (only into an empty database) ===
python -m app.seed_demo >> logs\setup.log 2>&1

echo === 5/5 Tests ===
python tests\test_app.py > logs\tests.log 2>&1
python tests\test_prepare.py >> logs\tests.log 2>&1
python tests\test_evaluate.py >> logs\tests.log 2>&1
type logs\tests.log | findstr /C:"passed" /C:"FAIL"

echo.
echo Setup finished. Start the app with run_app.bat
pause
