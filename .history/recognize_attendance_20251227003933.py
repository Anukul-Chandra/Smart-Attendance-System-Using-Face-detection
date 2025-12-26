import cv2
import os
import pickle
import csv
import numpy as np
from datetime import datetime
from deepface import DeepFace

from google_sheet import mark_attendance_sheet

# ---------------- CONFIG ----------------
EMBEDDINGS_DIR = "embeddings"
ATTENDANCE_FILE = "attendance.csv"

MODEL_NAME = "Facenet"
DETECTOR = "retinaface"
THRESHOLD = 0.9   # 🔥 correct threshold

# ---------------- LOAD EMBEDDINGS ----------------
db_embeddings = {}

for file in os.listdir(EMBEDDINGS_DIR):
    if file.endswith(".pkl"):
        name = file.replace(".pkl", "")
        with open(os.path.join(EMBEDDINGS_DIR, file), "rb") as f:
            db_embeddings[name] = np.array(pickle.load(f))

print("✅ Loaded users:", list(db_embeddings.keys()))

# ---------------- ATTENDANCE ----------------
def already_marked(name, today):
    if not os.path.exists(ATTENDANCE_FILE):
        return False
    with open(ATTENDANCE_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0] == name and row[1] == today:
                return True
    return False

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    if already_marked(name, today):
        print(f"⚠️ Already marked: {name}")
        mark_attendance_sheet(name, today, time_now)
        return

    file_exists = os.path.exists(ATTENDANCE_FILE)
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time", "Status"])
        writer.writerow([name, today, time_now, "Present"])

    mark_attendance_sheet(name, today, time_now)
    print(f"🟢 Attendance marked for {name}")

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
print("📷 Camera started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        reps = DeepFace.represent(
            img_path=frame,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True
        )

        live_emb = np.array(reps[0]["embedding"])

        best_match = "Unknown"
        min_dist = float("inf")

        for name, embeddings in db_embeddings.items():
            dists = np.linalg.norm(embeddings - live_emb, axis=1)
            avg_dist = np.mean(dists)

            if avg_dist < min_dist:
                min_dist = avg_dist
                best_match = name

        print(f"DEBUG → {best_match}, distance={min_dist}")

        if min_dist < THRESHOLD:
            mark_attendance(best_match)
            print("✅ Recognized:", best_match)
            break
        else:
            print("❌ Unknown face")

    except Exception:
        print("⚠️ No face detected")

cap.release()
cv2.destroyAllWindows()
