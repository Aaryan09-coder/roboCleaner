@echo off
echo ========================================
echo Nuclear Waste Cleaning Arm - Pose Detection
echo ========================================
echo.

REM Activate venv if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo Virtual environment activated
) else (
    echo WARNING: Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing requirements...
    pip install "numpy<2"
    pip install -r requirements.txt
)

echo.
echo Starting pose detection...
echo Press 'q' in the camera window to quit
echo.

python ml_model/yolo_fightingpose_detection.py

pause

