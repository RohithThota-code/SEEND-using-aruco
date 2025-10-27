#!/usr/bin/env python3
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

# =================== CONSTANTS ===================
LIDAR_PORT = '/dev/ttyUSB0'     # Lidar port
PIXHAWK_PORT = '/dev/ttyACM1'   # Pixhawk serial port
DDSM_PORT = '/dev/ttyACM0'
SERIAL_BAUDRATE = 115200
BAUDRATE = 57600
MIN_DISTANCE = 500              # mm obstacle threshold
TARGET_LAT = 17.3973234         # Example GPS target
TARGET_LON = 78.4899548
ALTITUDE = 0.0
WAYPOINT_REACHED_RADIUS = 1
FILE_NAME = "logged_coordinates.txt"
TURN_SPEED = 30
FORWARD_SPEED = 40

# ---- Vision / ArUco parameters ----
MARKER_LENGTH = 0.10  # meters
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
ARUCO_PARAMS = aruco.DetectorParameters_create()
CAMERA_MATRIX = np.loadtxt("camera_matrix.txt")
DIST_COEFFS = np.loadtxt("dist_coeffs.txt")

# ---- Globals ----
arrived = False
latest_scan = None
latest_servo1_value = None
latest_servo3_value = None
path = []
latest_tag = None
aruco_thread_running = True

# ---- Serial motor controller ----
ddsm_ser = serial.Serial(DDSM_PORT, baudrate=SERIAL_BAUDRATE)
ddsm_ser.setRTS(False)
ddsm_ser.setDTR(False)
print("[System] DDSM Connected")

# =================== FUNCTIONS ===================

def read_coordinates_from_file(filename):
    coords = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip().replace(" ", "")
                if not line:
                    continue
                try:
                    lat, lon = map(float, line.split(','))
                    coords.append((lat, lon))
                except ValueError:
                    print(f"[System] Skipping invalid line: {line}")
    except FileNotFoundError:
        print(f"[System] File not found: {filename}")
    return coords


def get_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = map(math.radians, [lat1, lat2])
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def lidar_thread_func(lidar):
    global latest_scan
    for scan in lidar.iter_scans():
        latest_scan = scan


def is_front_clear():
    global latest_scan
    scan_data = latest_scan
    if scan_data is None:
        return True
    for (_, angle, dist) in scan_data:
        if (angle >= 340 or angle <= 20) and 0 < dist < MIN_DISTANCE:
            return False
    return True


def is_left_clear():
    global latest_scan
    scan_data = latest_scan
    if scan_data is None:
        return True
    for (_, angle, dist) in scan_data:
        if (270 <= angle <= 340) and 0 < dist < (MIN_DISTANCE + 100):
            return False
    return True


def scale_servo_to_speed(servo_value):
    if servo_value is None:
        return 0
    return int((servo_value - 1500) / 500 * 100)


def avoid_obstacle():
    motor_control(0, 0)
    time.sleep(0.3)
    while not is_front_clear():
        print("[OBSTACLE] Avoiding...")
        motor_control(20, -20)
        time.sleep(0.5)
    motor_control(20, 20)
    time.sleep(1.5)


def follow_obstacle():
    while True:
        if not is_front_clear():
            print("[OBSTACLE] Turning right")
            motor_control(TURN_SPEED, -TURN_SPEED)
        elif not is_left_clear():
            print("[OBSTACLE] Moving forward")
            motor_control(FORWARD_SPEED, FORWARD_SPEED)
        else:
            break


def goto_position(vehicle, target_location):
    global latest_servo1_value, latest_servo3_value
    print(f"[Navigation] Moving to target: {target_location.lat}, {target_location.lon}")
    vehicle.simple_goto(target_location)
    while True:
        current_location = vehicle.location.global_relative_frame
        dist = get_haversine_distance(current_location.lat, current_location.lon,
                                      target_location.lat, target_location.lon)
        print(f"[Navigation] Distance to target: {dist:.2f} m")

        if dist <= WAYPOINT_REACHED_RADIUS:
            print("[Navigation] Target reached!")
            break

        if not is_front_clear():
            print("[Warning] Obstacle ahead!")
            follow_obstacle()

        s1, s3 = latest_servo1_value, latest_servo3_value
        motor_control(scale_servo_to_speed(s1), scale_servo_to_speed(s3))
        time.sleep(0.1)


def motor_control(left, right):
    command_r = {"T": 10010, "id": 2, "cmd": -right, "act": 3}
    command_l = {"T": 10010, "id": 1, "cmd": left, "act": 3}
    ddsm_ser.write((json.dumps(command_r) + '\n').encode())
    time.sleep(0.01)
    ddsm_ser.write((json.dumps(command_l) + '\n').encode())


# =============== ArUco vision thread ===============
def aruco_thread_func():
    global latest_tag, aruco_thread_running
    cap = cv2.VideoCapture(0)
    print("[Vision] ArUco camera started")

    while aruco_thread_running:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH,
                                                              CAMERA_MATRIX, DIST_COEFFS)
            distances = [np.linalg.norm(tvec[0]) for tvec in tvecs]
            nearest = int(np.argmin(distances))
            latest_tag = {
                "id": int(ids[nearest]),
                "x": tvecs[nearest][0][0],
                "y": tvecs[nearest][0][1],
                "z": tvecs[nearest][0][2]
            }
            aruco.drawDetectedMarkers(frame, corners)
            aruco.drawAxis(frame, CAMERA_MATRIX, DIST_COEFFS,
                           rvecs[nearest], tvecs[nearest], 0.05)
        else:
            latest_tag = None

        cv2.imshow("Aruco", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[Vision] ArUco stopped")


def vision_guided_navigation():
    """Simple proportional alignment to visible ArUco"""
    global latest_tag
    if latest_tag is None:
        motor_control(0, 0)
        print("[Vision] No tag visible")
        return False

    id_ = latest_tag["id"]
    x, y, z = latest_tag["x"], latest_tag["y"], latest_tag["z"]
    print(f"[Vision] Tag {id_}: x={x:.2f}, z={z:.2f}")

    if abs(x) > 0.05:
        if x > 0:
            motor_control(20, -20)
        else:
            motor_control(-20, 20)
    elif z > 0.4:
        motor_control(30, 30)
    else:
        motor_control(0, 0)
        print(f"[Vision] Reached tag {id_}")
        return True
    return False


# =================== MAIN ===================

def main():
    global latest_scan, path, aruco_thread_running
    path = read_coordinates_from_file(FILE_NAME)
    print(f"[System] Coordinates: {path}")

    print("[System] Starting LIDAR...")
    lidar = RPLidar(LIDAR_PORT)
    threading.Thread(target=lidar_thread_func, args=(lidar,), daemon=True).start()
    while latest_scan is None:
        print("[System] Waiting for LIDAR data...")
        time.sleep(1)
    print("[System] LIDAR active")

    # Start ArUco camera thread
    threading.Thread(target=aruco_thread_func, daemon=True).start()

    print("[System] Connecting to Pixhawk...")
    vehicle = connect(PIXHAWK_PORT, baud=BAUDRATE, wait_ready=False)
    print("[System] Pixhawk connected")

    @vehicle.on_message('SERVO_OUTPUT_RAW')
    def servo_listener(self, name, msg):
        global latest_servo1_value, latest_servo3_value
        latest_servo1_value = msg.servo1_raw
        latest_servo3_value = msg.servo3_raw

    print("[System] Arming vehicle...")
    vehicle.armed = True
    while not vehicle.armed:
        time.sleep(1)
    print("[System] Vehicle armed")

    print("[System] Setting GUIDED mode...")
    vehicle.mode = VehicleMode("GUIDED")
    while vehicle.mode.name != "GUIDED":
        time.sleep(1)
    print(f"[System] Mode: {vehicle.mode.name}")

    try:
        # -------- GPS Navigation Phase --------
        for i, coord in enumerate(path):
            loc = LocationGlobalRelative(coord[0], coord[1], ALTITUDE)
            print(f"[System] Waypoint {i+1}")
            goto_position(vehicle, loc)
            time.sleep(0.5)

        # -------- Vision-Guided Phase --------
        print("[System] Switching to vision-guided mode...")
        reached_ramp, reached_top, reached_door = False, False, False

        while True:
            if not reached_ramp:
                reached_ramp = vision_guided_navigation()
                if reached_ramp:
                    print("[System] ✅ Reached ramp entrance (Tag 1)")
                    time.sleep(1)

            elif not reached_top:
                reached_top = vision_guided_navigation()
                if reached_top:
                    print("[System] ✅ Reached top of ramp (Tag 2)")
                    time.sleep(1)

            elif not reached_door:
                reached_door = vision_guided_navigation()
                if reached_door:
                    print("[System] 🎯 Reached door (Tag 3)")
                    break

    except KeyboardInterrupt:
        print("[System] Keyboard interrupt")
        motor_control(0, 0)

    finally:
        aruco_thread_running = False
        motor_control(0, 0)
        vehicle.close()
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
        ddsm_ser.close()
        print("[System] Shutdown complete")

# =================== RUN ===================
if __name__ == "__main__":
    main()

