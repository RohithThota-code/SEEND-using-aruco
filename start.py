#!/usr/bin/env python3
"""
Jetson Nano rover: GPS + LIDAR baseline + ArUco vision-guided local navigation.
Combines your DDSM motor interface, Pixhawk DroneKit navigation and RPLidar scanning.
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
import sys

# =================== CONFIG ===================
VIDEO_STREAM_URL = '/dev/video0'                     # 0 for USB cam; or RTSP/IP stream string
LIDAR_PORT = '/dev/ttyUSB0'
PIXHAWK_PORT = '/dev/ttyACM1'
DDSM_PORT = '/dev/ttyACM0'
SERIAL_BAUDRATE = 115200
BAUDRATE = 57600

MIN_DISTANCE = 500                         # mm, obstacle threshold for LIDAR
WAYPOINT_REACHED_RADIUS = 1                # meters for GPS waypoint
ALTITUDE = 0.0

# Motor speed tuning (adjust for your rover)
TURN_SPEED = 30
FORWARD_SPEED = 40
SEARCH_TURN_SPEED = 20
SEARCH_TIMEOUT = 8.0                       # seconds to scan for next tag
DEAD_RECKON_FORWARD_SEC = 1.0              # fallback forward attempt (seconds)
MAX_SEARCH_ATTEMPTS = 3

# ArUco / camera
MARKER_LENGTH = 0.10   # meters (actual printed marker side)
# choose dictionary consistent with the markers you printed
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
ARUCO_PARAMS = aruco.DetectorParameters_create()

# Attempt to load calibration (if present) else use approximate default
CAMERA_MATRIX_FILE = "camera_matrix.txt"
DIST_COEFFS_FILE = "dist_coeffs.txt"

if os.path.exists(CAMERA_MATRIX_FILE) and os.path.exists(DIST_COEFFS_FILE):
    CAMERA_MATRIX = np.loadtxt(CAMERA_MATRIX_FILE)
    DIST_COEFFS = np.loadtxt(DIST_COEFFS_FILE)
    print("[Vision] Camera calibration loaded.")
else:
    # Fallback / approximate intrinsics (tweak for your camera/resolution)
    CAMERA_MATRIX = np.array([[800, 0, 320],
                              [0, 800, 240],
                              [0,   0,   1]], dtype=np.float32)
    DIST_COEFFS = np.zeros((5, 1))
    print("[Vision] Camera calibration files not found — using fallback intrinsics.")

# Tag sequence you expect (IDs printed on your markers)
TAG_SEQUENCE = [1, 2, 3]   # e.g., 1:ramp base, 2:ramp top, 3:door

# =================== GLOBALS ===================
latest_scan = None
latest_servo1_value = None
latest_servo3_value = None
latest_tag = None            # {"id": int, "x":..., "y":..., "z":...}
aruco_thread_running = True
path = []                    # optional GPS waypoint list

# =================== HARDWARE SETUP ===================
# DDSM serial
try:
    ddsm_ser = serial.Serial(DDSM_PORT, baudrate=SERIAL_BAUDRATE, timeout=0.2)
    ddsm_ser.setRTS(False)
    ddsm_ser.setDTR(False)
    print(f"[System] DDSM connected on {DDSM_PORT}")
except Exception as e:
    print(f"[ERROR] opening DDSM port {DDSM_PORT}: {e}")
    ddsm_ser = None

def motor_control(left, right):
    """Send left/right commands to DDSM. left/right = integer speed values."""
    global ddsm_ser
    if ddsm_ser is None:
        print(f"[Motor] Would send L={left} R={right} (no serial)")
        return
    cmd_r = {"T": 10010, "id": 2, "cmd": -int(right), "act": 3}
    cmd_l = {"T": 10010, "id": 1, "cmd": int(left),  "act": 3}
    try:
        ddsm_ser.write((json.dumps(cmd_r) + '\n').encode())
        time.sleep(0.01)
        ddsm_ser.write((json.dumps(cmd_l) + '\n').encode())
    except Exception as e:
        print(f"[Motor] Serial write error: {e}")

# =================== LIDAR THREAD ===================
def lidar_thread_func(lidar):
    global latest_scan
    try:
        for scan in lidar.iter_scans():
            latest_scan = scan
    except Exception as e:
        print(f"[LIDAR] Exception in lidar thread: {e}")

def is_front_clear():
    """Return False if obstacle in frontal sector within MIN_DISTANCE (mm)."""
    global latest_scan
    if latest_scan is None:
        return True
    for (_, angle, dist) in latest_scan:
        if (angle >= 340 or angle <= 20) and 0 < dist < MIN_DISTANCE:
            return False
    return True

# =================== ARUCO VISION THREAD ===================
def aruco_thread_func(vstream=VIDEO_STREAM_URL):
    """Continuously read camera, detect ArUco and store nearest tag in latest_tag."""
    global latest_tag, aruco_thread_running
    cap = cv2.VideoCapture(vstream)
    if not cap.isOpened():
        print("[Vision] ERROR: camera not opened.")
        aruco_thread_running = False
        return
    print("[Vision] Camera thread started.")
    while aruco_thread_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, CAMERA_MATRIX, DIST_COEFFS)
            distances = [np.linalg.norm(tvec[0]) for tvec in tvecs]
            nearest_idx = int(np.argmin(distances))
            latest_tag = {
                "id": int(ids.flatten()[nearest_idx]),
                "x": float(tvecs[nearest_idx][0][0]),
                "y": float(tvecs[nearest_idx][0][1]),
                "z": float(tvecs[nearest_idx][0][2])
            }
            # draw overlays
            aruco.drawDetectedMarkers(frame, corners, ids)
            aruco.drawAxis(frame, CAMERA_MATRIX, DIST_COEFFS, rvecs[nearest_idx], tvecs[nearest_idx], 0.05)
        else:
            latest_tag = None
        cv2.imshow("Aruco", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            # user asked to exit vision window
            break
    cap.release()
    cv2.destroyAllWindows()
    print("[Vision] Camera thread exiting.")

# =================== VISION HELPERS ===================
def vision_navigate_to_tag(target_id, align_thresh_x=0.05, stop_dist=1):
    """
    Move robot toward a *specific* target tag ID using latest_tag data.
    Returns True when reached (z <= stop_dist).
    When latest_tag is present but different ID, function will stop until target shows.
    """
    global latest_tag
    if latest_tag is None:
        motor_control(0, 0)
        print("[Vision] No tag visible.")
        return False
    if latest_tag["id"] != target_id:
        # we see some tag but it's not the one we want
        motor_control(0, 0)
        print(f"[Vision] Seen tag {latest_tag['id']}, waiting for target {target_id}.")
        return False

    x = latest_tag["x"]
    z = latest_tag["z"]
    print(f"[Vision] Target {target_id} seen: x={x:.3f} m, z={z:.3f} m")

    # basic alignment controller
    if abs(x) > align_thresh_x:
        if x > 0:
            motor_control(TURN_SPEED, -TURN_SPEED)   # turn right
        else:
            motor_control(-TURN_SPEED, TURN_SPEED)   # turn left
        return False
    # move forward if not too close and path is clear
    if z > stop_dist:
        if is_front_clear():
            motor_control(FORWARD_SPEED, FORWARD_SPEED)
        else:
            print("[Vision] Obstacle ahead - stopping.")
            motor_control(0, 0)
        return False
    # close enough
    motor_control(0, 0)
    print(f"[Vision] Reached target tag {target_id}.")
    return True

def search_for_next_tag(expected_id, timeout=SEARCH_TIMEOUT):
    """
    Rotate in place slowly searching for expected_id. Returns True if found.
    Stops rotation when found or on timeout.
    """
    print(f"[Vision] Scanning for tag {expected_id} (timeout {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        if latest_tag is not None and latest_tag["id"] == expected_id:
            print(f"[Vision] Found tag {expected_id} during scan.")
            motor_control(0, 0)
            return True
        # rotate in place slowly (right spin)
        motor_control(SEARCH_TURN_SPEED, -SEARCH_TURN_SPEED)
        time.sleep(0.15)
    motor_control(0, 0)
    print(f"[Vision] Scan timeout — tag {expected_id} not found.")
    return False

def dead_reckon_forward(seconds=DEAD_RECKON_FORWARD_SEC):
    """Move forward a small, fixed time to attempt to reveal next tag; careful: used as fallback."""
    print(f"[Vision] Dead-reckon forward for {seconds} s.")
    start = time.time()
    while time.time() - start < seconds:
        if not is_front_clear():
            print("[DeadReckon] Obstacle during dead-reckon, stopping.")
            motor_control(0, 0)
            return
        motor_control(FORWARD_SPEED, FORWARD_SPEED)
        time.sleep(0.05)
    motor_control(0, 0)

# =================== UTIL ===================
def read_coordinates_from_file(filename):
    coords = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.replace(" ", "").split(',')
                try:
                    lat = float(parts[0]); lon = float(parts[1])
                    coords.append((lat, lon))
                except Exception:
                    print(f"[System] Skipping invalid coordinate line: {s}")
    except FileNotFoundError:
        print(f"[System] Waypoint file not found: {filename}")
    return coords

def get_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# =================== MAIN ===================
def main():
    global latest_scan, path, aruco_thread_running, latest_servo1_value, latest_servo3_value

    # load path if present (optional)
    FILE_NAME = "logged_coordinates.txt"
    path = read_coordinates_from_file(FILE_NAME)
    print(f"[System] Loaded {len(path)} waypoints.")

    # start lidar thread
    try:
        lidar = RPLidar(LIDAR_PORT)
        threading.Thread(target=lidar_thread_func, args=(lidar,), daemon=True).start()
    except Exception as e:
        print(f"[LIDAR] Failed to start: {e}")
        lidar = None

    # wait for lidar data briefly (not mandatory)
    wait_count = 0
    while latest_scan is None and wait_count < 6:
        print("[System] Waiting for LIDAR data...")
        time.sleep(1)
        wait_count += 1

    # start vision thread
    threading.Thread(target=aruco_thread_func, daemon=True).start()
    time.sleep(0.5)  # allow camera to warm up

    # connect to pixhawk
    try:
        vehicle = connect(PIXHAWK_PORT, baud=BAUDRATE, wait_ready=False)
        print("[System] Connected to Pixhawk.")
    except Exception as e:
        print(f"[System] Pixhawk connection failed: {e}")
        vehicle = None

    # servo listener (if vehicle available)
    if vehicle:
        @vehicle.on_message('SERVO_OUTPUT_RAW')
        def servo_listener(self, name, msg):
            global latest_servo1_value, latest_servo3_value
            latest_servo1_value = getattr(msg, 'servo1_raw', None)
            latest_servo3_value = getattr(msg, 'servo3_raw', None)

        # arm and guided mode
        try:
            vehicle.armed = True
            while not vehicle.armed:
                print("[System] Waiting for arming...")
                time.sleep(1)
            vehicle.mode = VehicleMode("GUIDED")
            while vehicle.mode.name != "GUIDED":
                print("[System] Waiting for GUIDED mode...")
                time.sleep(1)
            print(f"[System] Vehicle mode: {vehicle.mode.name}")
        except Exception as e:
            print(f"[System] Vehicle setup error: {e}")

    try:
        # --- optional GPS-phase: drive to last GPS waypoint before vision phase ---
        if path and vehicle:
            for i, (lat, lon) in enumerate(path):
                print(f"[System] Going to waypoint {i+1}/{len(path)}: {lat},{lon}")
                target = LocationGlobalRelative(lat, lon, ALTITUDE)
                # simple goto (non-blocking) and monitor distance
                vehicle.simple_goto(target)
                while True:
                    cur = vehicle.location.global_relative_frame
                    dist = get_haversine_distance(cur.lat, cur.lon, lat, lon)
                    print(f"[Navigation] distance to waypoint: {dist:.1f} m")
                    if dist <= WAYPOINT_REACHED_RADIUS:
                        print("[Navigation] Waypoint reached.")
                        break
                    # handle lidar obstacles
                    if not is_front_clear():
                        print("[Navigation] LIDAR: obstacle ahead — following obstacle avoidance routine.")
                        follow_start = time.time()
                        # simple avoidance: rotate and move until clear
                        motor_control(TURN_SPEED, -TURN_SPEED)
                        time.sleep(0.6)
                        motor_control(FORWARD_SPEED, FORWARD_SPEED)
                        time.sleep(0.8)
                        motor_control(0,0)
                    # also support servo-based teleop fallback if present
                    if latest_servo1_value is not None:
                        left = int((latest_servo1_value - 1500) / 500 * 100)
                        right = int((latest_servo3_value - 1500) / 500 * 100) if latest_servo3_value is not None else left
                        motor_control(left, right)
                    time.sleep(0.4)
                motor_control(0,0)
                time.sleep(0.5)

        # --- Vision guided phase: follow tag sequence ---
        print("[System] Starting vision-guided phase.")
        for target_id in TAG_SEQUENCE:
            print(f"[System] Target tag: {target_id}")
            reached = False
            attempts = 0
            while not reached and attempts < MAX_SEARCH_ATTEMPTS:
                # 1) If currently visible and correct ID, try to approach
                if latest_tag is not None and latest_tag["id"] == target_id:
                    reached = vision_navigate_to_tag(target_id)
                    if reached:
                        break
                    # continue loop, vision_navigate_to_tag will command motors
                else:
                    # 2) search by rotating
                    found = search_for_next_tag(target_id, timeout=SEARCH_TIMEOUT)
                    if found:
                        # next loop iteration will approach
                        continue
                    # 3) if not found, attempt a short dead-reckon forward maneuver then retry
                    print(f"[System] Tag {target_id} not found after scan — dead-reckon attempt.")
                    dead_reckon_forward(DEAD_RECKON_FORWARD_SEC)
                    attempts += 1
            if not reached:
                # final chance: if still not reached, report and continue to next target or abort
                print(f"[System] Warning: failed to reach tag {target_id} after {MAX_SEARCH_ATTEMPTS} attempts. Continuing.")
            else:
                print(f"[System] Reached tag {target_id} successfully.")
            motor_control(0, 0)
            time.sleep(0.8)

        print("[System] Vision phase complete. Stopping.")
        motor_control(0, 0)
        time.sleep(0.5)

    except KeyboardInterrupt:
        print("[System] KeyboardInterrupt — stopping.")
        motor_control(0, 0)

    finally:
        # Shutdown / cleanup
        print("[System] Cleaning up...")
        aruco_thread_running = False
        motor_control(0, 0)
        try:
            if vehicle:
                vehicle.close()
        except Exception:
            pass
        try:
            if lidar:
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()
        except Exception:
            pass
        try:
            if ddsm_ser:
                ddsm_ser.close()
        except Exception:
            pass
        print("[System] Shutdown complete.")

if __name__ == "__main__":
    main()


