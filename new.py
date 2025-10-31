#!/usr/bin/env python3
"""
Jetson Nano Rover – GPS + LIDAR + ArUco Vision-Guided Navigation
Now supports the Jetson CSI camera via GStreamer (nvarguscamerasrc).
"""

from dronekit import connect, VehicleMode, LocationGlobalRelative
from rplidar import RPLidar
import math
import time
import threading
import serial
import json
import cv2
import cv2.aruco as aruco
import numpy as np
import os

# =================== CONFIG ===================
LIDAR_PORT = '/dev/ttyUSB0'
PIXHAWK_PORT = '/dev/ttyACM1'
DDSM_PORT = '/dev/ttyACM0'
SERIAL_BAUDRATE = 115200
BAUDRATE = 57600
MIN_DISTANCE = 500          # mm
ALTITUDE = 0.0
WAYPOINT_REACHED_RADIUS = 1
TURN_SPEED = 30
FORWARD_SPEED = 40
MARKER_LENGTH = 0.10        # meters (10 cm marker)
SEARCH_TIMEOUT = 8.0
DEAD_RECKON_FORWARD_SEC = 1.0
TAG_SEQUENCE = [1, 2, 3]    # tag IDs in navigation order

# ========== ARUCO CAMERA CALIBRATION ==========
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
ARUCO_PARAMS = aruco.DetectorParameters_create()

if os.path.exists("camera_matrix.txt") and os.path.exists("dist_coeffs.txt"):
    CAMERA_MATRIX = np.loadtxt("camera_matrix.txt")
    DIST_COEFFS = np.loadtxt("dist_coeffs.txt")
    print("[Vision] Loaded camera calibration.")
else:
    CAMERA_MATRIX = np.array([[800, 0, 320],
                              [0, 800, 240],
                              [0,   0,   1]], dtype=np.float32)
    DIST_COEFFS = np.zeros((5, 1))
    print("[Vision] Using fallback camera calibration.")

# ========== GLOBAL VARIABLES ==========
latest_scan = None
latest_servo1_value = None
latest_servo3_value = None
latest_tag = None
aruco_thread_running = True

# ========== SERIAL / MOTORS ==========
try:
    ddsm_ser = serial.Serial(DDSM_PORT, baudrate=SERIAL_BAUDRATE)
    ddsm_ser.setRTS(False)
    ddsm_ser.setDTR(False)
    print(f"[System] DDSM connected on {DDSM_PORT}")
except Exception as e:
    print(f"[ERROR] DDSM not connected: {e}")
    ddsm_ser = None


def motor_control(left, right):
    """Send JSON command to DDSM motor driver."""
    if ddsm_ser is None:
        print(f"[Motor] Skipped (no DDSM): L={left}, R={right}")
        return
    cmd_r = {"T": 10010, "id": 2, "cmd": -right, "act": 3}
    cmd_l = {"T": 10010, "id": 1, "cmd": left, "act": 3}
    ddsm_ser.write((json.dumps(cmd_r) + '\n').encode())
    time.sleep(0.01)
    ddsm_ser.write((json.dumps(cmd_l) + '\n').encode())


# ========== LIDAR THREAD ==========
def lidar_thread_func(lidar):
    global latest_scan
    for scan in lidar.iter_scans():
        latest_scan = scan


def is_front_clear():
    global latest_scan
    if latest_scan is None:
        return True
    for (_, angle, dist) in latest_scan:
        if (angle >= 340 or angle <= 20) and 0 < dist < MIN_DISTANCE:
            return False
    return True


# ========== CAMERA SETUP FOR JETSON ==========
def gstreamer_pipeline(
        capture_width=1280,
        capture_height=720,
        display_width=1280,
        display_height=720,
        framerate=30,
        flip_method=0):
    """Returns a GStreamer pipeline string for Jetson CSI camera."""
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width={capture_width}, height={capture_height}, "
        "format=NV12, framerate={}/1 ! ".format(framerate) +
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    )


def open_camera():
    """Try to open CSI camera first; fallback to USB."""
    gst = gstreamer_pipeline()
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[Vision] Using Jetson CSI camera.")
        return cap

    print("[Vision] Falling back to USB camera (/dev/video0).")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap

    print("❌ No camera detected.")
    return None


# ========== ARUCO THREAD ==========
def aruco_thread_func():
    global latest_tag, aruco_thread_running
    cap = open_camera()
    if cap is None:
        aruco_thread_running = False
        return

    while aruco_thread_running:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, CAMERA_MATRIX, DIST_COEFFS)
            distances = [np.linalg.norm(tvec[0]) for tvec in tvecs]
            nearest = int(np.argmin(distances))
            latest_tag = {
                "id": int(ids[nearest]),
                "x": float(tvecs[nearest][0][0]),
                "y": float(tvecs[nearest][0][1]),
                "z": float(tvecs[nearest][0][2])
            }
            aruco.drawDetectedMarkers(frame, corners)
            aruco.drawAxis(frame, CAMERA_MATRIX, DIST_COEFFS, rvecs[nearest], tvecs[nearest], 0.05)
        else:
            latest_tag = None

        cv2.imshow("Jetson Rover ArUco View", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Vision] Camera thread stopped.")


# ========== VISION NAVIGATION ==========
def vision_guided_navigation(target_id, stop_dist=1.0):
    """Approach a specific tag until within stop distance (in meters)."""
    global latest_tag
    if latest_tag is None:
        motor_control(0, 0)
        print("[Vision] No tag visible.")
        return False

    id_ = latest_tag["id"]
    x, z = latest_tag["x"], latest_tag["z"]

    if id_ != target_id:
        motor_control(0, 0)
        print(f"[Vision] Saw tag {id_}, looking for {target_id}.")
        return False

    print(f"[Vision] Tag {id_} @ x={x:.2f} m, z={z:.2f} m")

    if abs(x) > 0.05:
        # Adjust orientation
        if x > 0:
            motor_control(TURN_SPEED, -TURN_SPEED)
        else:
            motor_control(-TURN_SPEED, TURN_SPEED)
    elif z > stop_dist:
        if is_front_clear():
            motor_control(FORWARD_SPEED, FORWARD_SPEED)
        else:
            print("[Vision] Obstacle ahead.")
            motor_control(0, 0)
    else:
        motor_control(0, 0)
        print(f"[Vision] Reached tag {id_}")
        return True
    return False


def search_for_next_tag(expected_id, timeout=SEARCH_TIMEOUT):
    """Rotate to find the expected tag."""
    print(f"[Vision] Searching for tag {expected_id} (timeout={timeout}s)")
    start = time.time()
    while time.time() - start < timeout:
        if latest_tag is not None and latest_tag["id"] == expected_id:
            print(f"[Vision] Found tag {expected_id}")
            motor_control(0, 0)
            return True
        motor_control(20, -20)
        time.sleep(0.2)
    motor_control(0, 0)
    print("[Vision] Scan timeout.")
    return False


# ========== MAIN ==========
def main():
    global latest_scan, aruco_thread_running

    print("[System] Starting LIDAR...")
    lidar = RPLidar(LIDAR_PORT)
    threading.Thread(target=lidar_thread_func, args=(lidar,), daemon=True).start()
    while latest_scan is None:
        print("[System] Waiting for LIDAR data...")
        time.sleep(1)
    print("[System] LIDAR running.")

    # Start camera thread
    threading.Thread(target=aruco_thread_func, daemon=True).start()
    time.sleep(2)

    print("[System] Connecting to Pixhawk...")
    vehicle = connect(PIXHAWK_PORT, baud=BAUDRATE, wait_ready=False)
    print("[System] Pixhawk connected.")

    print("[System] Arming vehicle...")
    vehicle.armed = True
    while not vehicle.armed:
        time.sleep(1)
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != "GUIDED":
        time.sleep(1)
    print(f"[System] Mode: {vehicle.mode.name}")

    try:
        print("[System] Starting vision-guided phase...")
        for target_id in TAG_SEQUENCE:
            print(f"[System] Looking for tag {target_id}")
            reached = False
            attempts = 0
            while not reached and attempts < 3:
                if latest_tag is not None and latest_tag["id"] == target_id:
                    reached = vision_guided_navigation(target_id)
                    if reached:
                        break
                else:
                    found = search_for_next_tag(target_id)
                    if not found:
                        print("[System] Moving forward briefly to recheck.")
                        motor_control(FORWARD_SPEED, FORWARD_SPEED)
                        time.sleep(DEAD_RECKON_FORWARD_SEC)
                        motor_control(0, 0)
                        attempts += 1
            if reached:
                print(f"[System] ✅ Tag {target_id} reached.")
                time.sleep(1)
            else:
                print(f"[System] ⚠️ Could not find tag {target_id} after retries.")
                time.sleep(1)

        print("[System] Mission complete.")
        motor_control(0, 0)

    except KeyboardInterrupt:
        print("[System] Stopping manually.")
        motor_control(0, 0)

    finally:
        print("[System] Shutting down...")
        aruco_thread_running = False
        motor_control(0, 0)
        vehicle.close()
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
        if ddsm_ser:
            ddsm_ser.close()
        print("[System] Shutdown complete.")


# ========== RUN ==========
if __name__ == "__main__":
    main()
