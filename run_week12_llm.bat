@echo off
REM Week 12: one-click LLM run. Double-click this file, or run it from a terminal.
REM Safe to run again: finished LLM answers are cached, so it continues where it stopped.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
if errorlevel 1 ( echo Could not activate .venv & pause & exit /b 1 )
if not exist results mkdir results

echo === 1/5 Commit the week 10 results and the week 12 script (no push) ===
git add README.md results\week10_test_metrics.csv results\week10_check_val_metrics.csv src\llm_estimate.py docs\week12_llm.md run_week12_llm.bat
git commit -m "Week 10 final test evaluation; week 12 LLM estimation script" -m "SBERT+kNN (validation winner) meets the criterion 10/16 but its test macro MAE 2.656 is above the week 9 RF 2.536, caused by moodle estimation drift. Application keeps the week 9 RF. Adds llm_estimate.py (zero-shot and retrieval few-shot via Ollama)."

echo.
echo === 2/5 Ollama ===
where ollama >nul 2>nul
if errorlevel 1 (
  echo Ollama not found - installing with winget ...
  winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
  set "PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%"
)
where ollama >nul 2>nul
if errorlevel 1 (
  echo.
  echo Ollama could not be installed automatically.
  echo Install it from https://ollama.com/download and run this file again.
  pause & exit /b 1
)
ollama list >nul 2>nul
if errorlevel 1 (
  echo Starting the Ollama server ...
  start "ollama" /min ollama serve
  timeout /t 8 /nobreak >nul
)

echo.
echo === 3/5 Download the model (about 4.7 GB, skipped if already there) ===
ollama pull qwen2.5:7b
if errorlevel 1 ( echo Model download failed - check the internet connection and run again. & pause & exit /b 1 )

echo.
echo === 4/5 Pipeline check: 3 issues per project, validation split ===
powershell -NoProfile -Command "python -u src\llm_estimate.py --split val --n 3 2>&1 | Tee-Object -FilePath results\week12_check_run.log"

echo.
echo === 5/5 Real run: 20 issues per project, TEST split (can take several hours) ===
powershell -NoProfile -Command "python -u src\llm_estimate.py --split test --n 20 2>&1 | Tee-Object -FilePath results\week12_test_run.log"

echo.
echo Finished. Results are in the results folder (week12_*). You can close this window.
pause
