#!/usr/bin/env python
"""
Step-by-step setup and execution script for Nuclear Waste Cleaning Arm
Run this script to set up everything and test the pose detection.
"""
import sys
import os
import subprocess

def print_step(step_num, description):
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {description}")
    print('='*60)

def run_command(cmd, check=True):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)

def main():
    print("\n" + "="*60)
    print("NUCLEAR WASTE CLEANING ARM - SETUP & TEST")
    print("="*60)
    
    # Step 1: Check Python version
    print_step(1, "Checking Python version")
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("ERROR: Python 3.8+ required")
        return
    print("✓ Python version OK")
    
    # Step 2: Check if venv exists, create if not
    print_step(2, "Setting up virtual environment")
    if os.path.exists("venv"):
        print("✓ Virtual environment already exists")
    else:
        print("Creating virtual environment...")
        success, output = run_command(f"{sys.executable} -m venv venv")
        if success:
            print("✓ Virtual environment created")
        else:
            print(f"ERROR creating venv: {output}")
            return
    
    # Step 3: Determine activation script
    print_step(3, "Activating virtual environment")
    if sys.platform == "win32":
        activate_script = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:
        activate_script = "venv/bin/activate"
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    print(f"Using: {python_cmd}")
    
    # Step 4: Upgrade pip
    print_step(4, "Upgrading pip")
    success, output = run_command(f"{pip_cmd} install --upgrade pip")
    if success:
        print("✓ pip upgraded")
    else:
        print(f"Warning: {output}")
    
    # Step 5: Install numpy < 2 first (to avoid conflicts)
    print_step(5, "Installing NumPy < 2.0 (required for compatibility)")
    success, output = run_command(f"{pip_cmd} install 'numpy<2'")
    if success:
        print("✓ NumPy installed")
    else:
        print(f"Warning: {output}")
    
    # Step 6: Install other requirements
    print_step(6, "Installing requirements")
    success, output = run_command(f"{pip_cmd} install -r requirements.txt")
    if success:
        print("✓ Requirements installed")
    else:
        print(f"Warning: Some packages may have issues: {output}")
    
    # Step 7: Verify key imports
    print_step(7, "Verifying key imports")
    test_script = """
import sys
errors = []
try:
    import numpy
    print(f"✓ NumPy {numpy.__version__}")
except Exception as e:
    errors.append(f"NumPy: {e}")

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except Exception as e:
    errors.append(f"OpenCV: {e}")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except Exception as e:
    errors.append(f"PyTorch: {e}")

try:
    from ultralytics import YOLO
    print("✓ Ultralytics YOLO")
except Exception as e:
    errors.append(f"Ultralytics: {e}")

if errors:
    print("\\nErrors:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
"""
    success, output = run_command(f'{python_cmd} -c "{test_script}"')
    if success:
        print(output)
        print("✓ All key imports successful")
    else:
        print("ERROR in imports:")
        print(output)
        return
    
    # Step 8: Check model file
    print_step(8, "Checking model file")
    model_path = "model_assets/yolo11n-pose.pt"
    if os.path.exists(model_path):
        print(f"✓ Model file found: {model_path}")
    else:
        print(f"ERROR: Model file not found at {model_path}")
        print("Please ensure the model file exists")
        return
    
    # Step 9: Test pose detector initialization
    print_step(9, "Testing pose detector initialization")
    test_detector = """
import sys
sys.path.insert(0, '.')
try:
    from ml_model.yolo_fightingpose_detection import ZonePoseDetector
    print("Importing detector...")
    detector = ZonePoseDetector()
    print("✓ Detector initialized successfully!")
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
"""
    success, output = run_command(f'{python_cmd} -c "{test_detector}"')
    if success:
        print(output)
        print("✓ Detector test passed")
    else:
        print("ERROR in detector initialization:")
        print(output)
        return
    
    # Step 10: Test camera
    print_step(10, "Testing camera access")
    test_camera = """
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"✓ Camera working! Frame shape: {frame.shape}")
    else:
        print("⚠ Camera opened but cannot read frames")
    cap.release()
else:
    print("⚠ Camera not accessible (this is OK if no camera is connected)")
"""
    success, output = run_command(f'{python_cmd} -c "{test_camera}"')
    print(output)
    
    # Final step: Run the pose detection
    print_step(11, "READY TO RUN!")
    print("\nTo run the pose detection, use one of these commands:")
    print(f"\n  {python_cmd} ml_model/yolo_fightingpose_detection.py")
    print("\nOr activate the venv first:")
    if sys.platform == "win32":
        print("  venv\\Scripts\\activate")
    else:
        print("  source venv/bin/activate")
    print("  python ml_model/yolo_fightingpose_detection.py")
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    
    # Ask if user wants to run now
    response = input("\nDo you want to run the pose detection now? (y/n): ").strip().lower()
    if response == 'y':
        print("\nStarting pose detection...")
        print("Press 'q' in the camera window to quit\n")
        os.system(f'{python_cmd} ml_model/yolo_fightingpose_detection.py')
    else:
        print("\nYou can run it later with:")
        print(f"  {python_cmd} ml_model/yolo_fightingpose_detection.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()

