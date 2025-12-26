import cv2
import os
import numpy as np
from datetime import datetime
from deepface import DeepFace
import csv

# ---------------- CONFIG ----------------
REFERENCE_IMG = "face_db/anukul chandra/img1.jpg"
ATTENDANCE_FILE = "attendance.csv"

MODEL_NAME = "Facenet"
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.40   # strict (VERY IMPORTANT)

# ---------------- ATTENDANCE ----------------
def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == name and row[1] == today:
                    print("⚠️ Attendance already marked")
                    return

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["Name", "Date", "Time", "Status"])
        writer.writerow([name, today, time_now, "Present"])

    print(f"🟢 Attendance marked for {name}")

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print("📷 Camera started. Look at the camera...")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # DeepFace auto face detection + verification
        result = DeepFace.verify(
            img1_path=frame,
            img2_path=REFERENCE_IMG,
            model_name=MODEL_NAME,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=True
        )

        distance = result["distance"]
        verified = result["verified"]

        print("DEBUG → distance:", distance)

        if verified and distance < THRESHOLD:
            name = "anukul chandra"
            mark_attendance(name)
            label = f"{name} ✅"
            color = (0, 255, 0)

            cv2.putText(frame, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("Smart Attendance", frame)
            cv2.waitKey(1500)
            break

        else:
            label = "Unknown ❌"
            color = (0, 0, 255)

    except Exception:
        label = "No Face ❌"
        color = (0, 0, 255)

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Smart Attendance", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
print("✅ Camera closed")
