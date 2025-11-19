#!/usr/bin/env python3
"""
Simple test script to exercise all four servo axes on the ESP32 controller.

Usage:
    python client/test_send_angles.py --host 192.168.1.100 --port 8000

Sequence overview:
    1. Move base left/right
    2. Move shoulder forward/backward
    3. Move elbow up/down
    4. Open/close gripper
"""
import socket
import json
import time
import argparse


def send_command(s, cmd):
    s.sendall((json.dumps(cmd) + "\n").encode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="192.168.1.100")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    addr = (args.host, args.port)
    print(f"Connecting to {addr}...")
    try:
        s = socket.create_connection(addr, timeout=5)
    except Exception as e:
        print("Failed to connect:", e)
        return

    try:
        def send_pose(s1, s2, s3, s4, delay=1.0, label=""):
            cmd = {
                "type": "servo",
                "servo1": int(s1),
                "servo2": int(s2),
                "servo3": int(s3),
                "servo4": int(s4),
            }
            if label:
                print(f"{label}: {cmd}")
            else:
                print("Sending:", cmd)
            send_command(s, cmd)
            time.sleep(delay)

        neutral = (90, 90, 90, 90)
        send_pose(*neutral, label="Neutral pose")

        # Base sweep
        for angle in (30, 150, 90):
            send_pose(angle, neutral[1], neutral[2], neutral[3], label="Base sweep")

        # Forward/backward
        for angle in (40, 140, 90):
            send_pose(neutral[0], angle, neutral[2], neutral[3], label="Forward/backward")

        # Up/down
        for angle in (30, 150, 90):
            send_pose(neutral[0], neutral[1], angle, neutral[3], label="Up/down")

        # Grip open/close
        for g in (0, 180, 90):
            send_pose(neutral[0], neutral[1], neutral[2], g, delay=0.8, label="Grip")

        print("Sequence complete")
    finally:
        try:
            s.close()
        except:
            pass


if __name__ == "__main__":
    main()
