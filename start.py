#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np
import serial
import time

# ---------------------- CONFIG ---------------------- #
SERIAL_PORT = '/dev/ttyUSB0'  # change if different
BAUD_RATE = 9600

MARKER_LENGTH = 0.10  # meters
TARGET_SEQUENCE = [1, 2, 3]  # ramp, top, door

ALIGN_THRESHOLD_X = 0.05  # meters left/right
APPROACH_DISTANCE_Z = 0.40  # meters forward
REACHED_DISTANCE = 0.25     # stop threshold
# ---------------------------------------------------- #

# Serial connection to Arduino
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"✅ Connected to motor controller at {SERIAL_PORT}")
except Exception as e:
    print(f"⚠️ Could not connect to serial port: {e}")
    ser = None

def move_forward(): send_cmd('F')
def turn_left(): send_cmd('L')
def turn_right(): send_cmd('R')
def stop(): send_cmd('S')

def send_cmd(cmd):
    if ser:
        ser.write(cmd.encode())
        time.sleep(0.05)

# -------------------- Camera setup -------------------- #
cap = cv2.VideoCapture(0)
dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
parameters = aruco.DetectorParameters_create()

camera_matrix = np.loadtxt("camera_matrix.txt")
dist_coeffs = np.loadtxt("dist_coeffs.txt")

# -------------------- State machine -------------------- #
state = "SEARCH_RAMP"
current_target_idx = 0

def detect_tags(frame):
    """Detect ArUco tags and return {id: (x, y, z)}"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)

    tag_positions = {}
    if ids is not None:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)
        for i, id_ in enumerate(ids.flatten()):
            tvec = tvecs[i][0]
            tag_positions[id_] = (tvec[0], tvec[1], tvec[2])
            aruco.drawDetectedMarkers(frame, corners)
            aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvecs[i], tvec, 0.05)
    return tag_positions

def control_to_tag(x, z):
    """Align to and approach a tag"""
    if abs(x) > ALIGN_THRESHOLD_X:
        if x > 0:
            turn_right()
            print("↪️ Turning right")
        else:
            turn_left()
            print("↩️ Turning left")
    elif z > APPROACH_DISTANCE_Z:
        move_forward()
        print("⬆️ Moving forward")
    else:
        stop()
        print("🟢 Close to tag")

print("🚗 Starting ramp navigation...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera frame not captured")
        break

    tags = detect_tags(frame)
    current_target = TARGET_SEQUENCE[current_target_idx]

    if state == "SEARCH_RAMP":
        print(f"Searching for ramp tag ({current_target})...")
        if current_target in tags:
            state = "ALIGN_TO_RAMP"

    elif state == "ALIGN_TO_RAMP":
        x, y, z = tags.get(current_target, (None, None, None))
        if x is not None:
            control_to_tag(x, z)
            if z < REACHED_DISTANCE:
                stop()
                print(f"✅ Reached tag {current_target}")
                current_target_idx += 1
                if current_target_idx >= len(TARGET_SEQUENCE):
                    state = "DONE"
                else:
                    state = "CLIMB_RAMP"
        else:
            state = "SEARCH_RAMP"

    elif state == "CLIMB_RAMP":
        print("🧗 Climbing ramp...")
        next_target = TARGET_SEQUENCE[current_target_idx]
        if next_target in tags:
            x, y, z = tags[next_target]
            control_to_tag(x, z)
            if z < REACHED_DISTANCE:
                stop()
                print(f"✅ Reached top of ramp (Tag {next_target})")
                current_target_idx += 1
                if current_target_idx >= len(TARGET_SEQUENCE):
                    state = "DONE"
                else:
                    state = "APPROACH_DOOR"
        else:
            move_forward()

    elif state == "APPROACH_DOOR":
        print("🚪 Approaching door...")
        next_target = TARGET_SEQUENCE[current_target_idx]
        if next_target in tags:
            x, y, z = tags[next_target]
            control_to_tag(x, z)
            if z < REACHED_DISTANCE:
                stop()
                print(f"🎯 Reached door (Tag {next_target})")
                state = "DONE"
        else:
            move_forward()

    elif state == "DONE":
        stop()
        print("🏁 Mission complete — rover stopped at the door.")
        break

    cv2.imshow("Rover Vision", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
        break

cap.release()
cv2.destroyAllWindows()
stop()
