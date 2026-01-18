@echo off
echo ========================================
echo   Sign Language Translator - Streamlit
echo ========================================
echo.

REM Check if venv exists
if not exist "venv" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
)

REM Activate venv
echo [2/3] Activating environment...
call venv\Scripts\activate.bat

REM Install requirements
echo [3/3] Installing dependencies...
pip install -r requirements.txt --quiet

echo.
echo ========================================
echo   Starting Streamlit App...
echo   Open: http://localhost:8501
echo ========================================
echo.

streamlit run app.py

pause
