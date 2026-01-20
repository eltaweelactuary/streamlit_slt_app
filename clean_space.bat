@echo off
echo ========================================
echo   Cleaning Disk Space (Uninstall)
echo ========================================
echo.

cd /d "%~dp0"

echo [WARNING] This will delete the 'venv' folder and free up space.
echo The app will NOT run locally after this until you reinstall.
echo.
echo Make sure you have copied 'psl_classifier.pkl' if you need it!
echo.
pause

echo.
echo [1/3] Killing processes...
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM streamlit.exe /T >nul 2>&1

echo [2/3] Deleting virtual environment (This frees up ~2GB)...
if exist "venv" (
    rmdir /s /q venv
)
if exist "__pycache__" (
    rmdir /s /q __pycache__
)

echo [3/3] Done! Space reclaimed.
echo.
pause
