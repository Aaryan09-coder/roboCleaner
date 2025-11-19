# Nuclear Waste Cleaning Robotic Arm

A real-time pose-controlled robotic arm system for nuclear waste cleaning operations. This project uses [YOLO V11 Pose Detection](https://github.com/ultralytics/ultralytics) to translate human arm movements into precise servo motor commands for a 3-DOF (Degree of Freedom) robotic arm.

## Description

This system controls a robotic arm with three servo motors (shoulder, elbow, and wrist) using real-time human pose detection. The arm mimics the operator's arm movements, allowing for intuitive control of nuclear waste cleaning operations. The system uses YOLO pose estimation to track arm angles and translates them into servo commands in real-time.

## Gallery
![IMG_4483](https://github.com/user-attachments/assets/f2ba38a2-d61a-440a-8a33-d0f62f6f3232)

![19C21752-C7D1-428F-B070-E98F84A3DB98_4_5005_c](https://github.com/user-attachments/assets/cdbb7106-7beb-42ac-8e96-8922efde17bd)

![image](https://github.com/user-attachments/assets/fb3f225c-a0a8-469a-aaa4-b4840ad9dbf2)

![image](https://github.com/user-attachments/assets/d8341888-6c11-46dc-a18b-117827af595b)

## How It Works

1. **Pose Detection (YOLO + OpenCV)**: The system uses YOLO pose estimation to track the operator's arm movements in real time.
2. **Angle Calculation**: The AI calculates shoulder, elbow, and wrist angles from detected keypoints.
3. **Servo Control**: Calculated angles are mapped to servo positions (0-180 degrees) for each joint.
4. **ESP32 Controller**: The firmware uploaded to the ESP32 controls three servo motors (shoulder, elbow, wrist) connected to GPIO pins.
5. **Real-time Control**: Detected arm angles are sent to the ESP32 controller via WiFi sockets, allowing the robotic arm to mimic the operator's movements for precise nuclear waste cleaning operations.

## Tech/Hardware Stack

-   Python, PyQt5 → Graphical user interface (UI) for system control.
-   YOLO (Ultralytics) → Real-time pose detection and arm angle calculation.
-   OpenCV → Camera feed processing and visualization.
-   C++, ESP32, Servo Motors → Robotic arm motion control (3 servos: shoulder, elbow, wrist).

## Getting Started

1. Install conda on your local machine through this [link](https://docs.anaconda.com/anaconda/install/) or through the OS package manager provided.
2. For Apple Silicon based chips:
    - Create your conda environment using `conda create -n ultralytics-env python=3.12 -y`.
3. For AMD/Intel based chips:
    - Create your conda enviroment using `CONDA_SUBDIR=osx-arm64 conda create -n ultralytics-env python=3.12 -y`.
4. Enter the conda environment by entering `conda activate ultralytics-env`.
5. Install necessary python libraries in conda environment using `pip install -r requirements.txt`.
6. Run the project inside the UI folder using `python3 finalUI.py`.

# Running the Program
1. Ensure you have the firmware in the /controller PlatformIO project installed on the ESP32 Microcontroller
2. Power the ESP32 and connect your laptop to its WiFi network (ESP32_Servo_Control_Blue)
3. Connect three servo motors to GPIO pins:
   - Shoulder servo → GPIO 18
   - Elbow servo → GPIO 19
   - Wrist servo → GPIO 22
4. Run the program:
   - For pose control: `python client/main.py` and select "yolo" mode
   - For keyboard control: `python client/main.py` and select "keyboard" mode
   - Or use the UI: Navigate to ./UI directory and run `python finalUI.py`

## Notes

-   For Macbook Intel Chip you need to install `pip install "numpy<2"` manually.
-   For any Apple Silicon Chips you cannot access the keyboard controller.

## Resources

-   [Ultralytics Coco Pose Estimation](https://docs.ultralytics.com/datasets/pose/coco/)
-   [Ultralytics Pose Estimation Tasks](https://docs.ultralytics.com/tasks/pose/)

## Team Members

1. Aaryan Waghmare
2. Khush Kothari
3. Tanvi Shinde
