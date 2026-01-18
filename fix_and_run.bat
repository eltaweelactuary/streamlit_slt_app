@echo off
echo ========================================
echo   Fixing & Starting Streamlit App
echo ========================================
echo.

REM Check if venv exists
if not exist "venv" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
)

REM Activate venv
echo [2/4] Activating environment...
call venv\Scripts\activate.bat

REM Force reinstall critical libs
echo [3/4] Fixing Dependencies (This might take a minute)...
echo     - Upgrading pip...
python -m pip install --upgrade pip --quiet
echo     - Reinstalling MediaPipe & Protobuf...
pip uninstall -y mediapipe protobuf --quiet
pip install mediapipe protobuf --no-cache-dir --quiet
echo     - Installing other requirements...
pip install -r requirements.txt --quiet

echo.
echo ========================================
echo   Starting Streamlit App...
echo   Open: http://localhost:8501
echo ========================================
echo.

streamlit run app.py

pause
