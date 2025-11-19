@echo off
echo ========================================
echo Setting up Virtual Environment
echo ========================================
echo.

echo Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create venv
    pause
    exit /b 1
)

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 4: Installing NumPy < 2.0...
pip install "numpy<2"

echo.
echo Step 5: Installing requirements...
pip install -r requirements.txt

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To run pose detection, use: run_pose_detection.bat
echo Or manually: venv\Scripts\activate then python ml_model/yolo_fightingpose_detection.py
echo.
pause

