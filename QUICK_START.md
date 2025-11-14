# Quick Start Guide - Nuclear Waste Cleaning Arm

## Step-by-Step Setup (Copy and paste these commands one by one)

### Step 1: Create Virtual Environment
```powershell
python -m venv venv
```

### Step 2: Activate Virtual Environment
```powershell
venv\Scripts\activate
```

### Step 3: Upgrade pip
```powershell
python -m pip install --upgrade pip
```

### Step 4: Install NumPy < 2.0 (IMPORTANT - do this first!)
```powershell
pip install "numpy<2"
```

### Step 5: Install all requirements
```powershell
pip install -r requirements.txt
```

### Step 6: Test the pose detection
```powershell
python ml_model/yolo_fightingpose_detection.py
```

---

## If you get errors:

### Error: NumPy version issue
```powershell
pip uninstall numpy -y
pip install "numpy==1.26.4"
```

### Error: Model file not found
Make sure `model_assets/yolo11n-pose.pt` exists in the project root.

### Error: Camera not working
The script will still run, but you won't see video. Make sure your camera is connected and not used by another app.

---

## That's it! 

Once Step 6 runs successfully, you should see:
- A camera window with pose detection
- Real-time angle values displayed
- Press 'q' to quit

