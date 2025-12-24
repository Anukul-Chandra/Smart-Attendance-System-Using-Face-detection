import cv2
import os
import pickle
import csv
import numpy as np
from datetime import datetime
from deepface import DeepFace

# ✅ Google Sheet helper import
from google_sheet import mark_attendance_sheet

# ---------------- PATHS ----------------
EMBEDDINGS_DIR = "embeddings"
ATTENDANCE_FILE = "attendance.csv"

# ---------------- LOAD EMBEDDINGS ----------------
db_embeddings = {}

for file in os.listdir(EMBEDDINGS_DIR):
    if file.endswith(".pkl"):
        name = file.replace(".pkl", "")
        with open(os.path.join(EMBEDDINGS_DIR, file), "rb") as f:
            db_embeddings[name] = pickle.load(f)

print("✅ Loaded users:", list(db_embeddings.keys()))

# ---------------- ATTENDANCE FUNCTIONS ----------------
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
        print(f"⚠️ Attendance already marked for {name}")
        return

    # ---- CSV SAVE ----
    file_exists = os.path.exists(ATTENDANCE_FILE)
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time", "Status"])
        writer.writerow([name, today, time_now, "Present"])

    # ---- GOOGLE SHEET SAVE ----
    try:
        mark_attendance_sheet(name, today, time_now)
        print(f"🟢 Attendance marked for {name} (CSV + Sheet)")
    except Exception as e:
        print("❌ Google Sheet error:", e)

# ---------------- FACE DETECTOR ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access camera")
    exit()

print("📷 Camera started. Show your face...")

THRESHOLD = 10  # Facenet distance threshold

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        try:
            rep = DeepFace.represent(
                img_path=face_img,
                model_name="Facenet",
                enforce_detection=False
            )
            live_embedding = np.array(rep[0]["embedding"])
        except Exception:
            continue

        name_found = "Unknown"
        min_dist = float("inf")

        for name, embeddings in db_embeddings.items():
            for emb in embeddings:
                dist = np.linalg.norm(live_embedding - np.array(emb))
                if dist < min_dist:
                    min_dist = dist
                    name_found = name

        # ---------- RECOGNITION RESULT ----------
        if min_dist < THRESHOLD:
            mark_attendance(name_found)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                name_found,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            cv2.imshow("Attendance System", frame)
            cv2.waitKey(1200)

            cap.release()
            cv2.destroyAllWindows()
            print("✅ Attendance complete. Camera closed automatically.")
            exit()

        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(
                frame,
                "Unknown",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

    cv2.imshow("Attendance System", frame)

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
