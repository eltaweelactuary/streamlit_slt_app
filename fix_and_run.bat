@echo off
echo ========================================
echo   Fixing & Starting Streamlit App (Final Recovery)
echo ========================================
echo.

cd /d "%~dp0"

REM 0. KILL RUNNING PROCESSES (Releases file locks)
echo [0/7] Killing lingering Python processes...
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM streamlit.exe /T >nul 2>&1

REM 1. CLEANUP (Critical for Space Issue)
if exist "venv" (
    echo [1/7] Formatting Corrupted Environment...
    rmdir /s /q venv
)

REM 2. Create Fresh Venv
echo [2/7] Creating Clean Environment...
python -m venv venv

REM 3. Define executables
set PYTHON_EXE=venv\Scripts\python.exe
set PIP_EXE=venv\Scripts\pip.exe

REM 4. Upgrade Pip
echo [3/7] Upgrading Pip...
"%PYTHON_EXE%" -m pip install --upgrade pip --quiet

REM 5. Pre-Install Torch Correctly (Avoid Conflict)
echo [4/7] Installing Torch CPU (Lightweight)...
"%PIP_EXE%" install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir --quiet

REM 6. Install Rest
echo [5/7] Installing App Dependencies...
"%PIP_EXE%" install -r requirements.txt --quiet --no-deps

REM 7. Manual Link Fix (Include Blinker & missing deps)
echo [6/7] Linking Libraries (and missing deps)...
"%PIP_EXE%" install mediapipe protobuf sign-language-translator scikit-learn streamlit opencv-python-headless blinker cachetools --no-deps --quiet

echo.
echo ========================================
echo   Starting Streamlit App...
echo   Open: http://localhost:8501
echo ========================================
echo.

"%PYTHON_EXE%" -m streamlit run app.py

pause
