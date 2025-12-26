import cv2
import os
import csv
from datetime import datetime
from deepface import DeepFace

from google_sheet import mark_attendance_sheet

# ---------------- CONFIG ----------------
FACE_DB_DIR = "face_db/anukul chandra"   # enrolled user folder
REFERENCE_IMG = os.path.join(FACE_DB_DIR, "img1.jpg")  # reference face
ATTENDANCE_FILE = "attendance.csv"

VERIFY_MODEL = "Facenet"
VERIFY_METRIC = "cosine"
VERIFY_THRESHOLD = 0.40   # 0.35–0.45 recommended for FaceNet + cosine

# ---------------- ATTENDANCE HELPERS ----------------
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
        print(f"⚠️ Attendance already marked for {name}, syncing to sheet")
        try:
            mark_attendance_sheet(name, today, time_now)
        except Exception as e:
            print("❌ Google Sheet error:", e)
        return

    file_exists = os.path.exists(ATTENDANCE_FILE)
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time", "Status"])
        writer.writerow([name, today, time_now, "Present"])

    try:
        mark_attendance_sheet(name, today, time_now)
        print(f"🟢 Attendance marked for {name}")
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

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(90, 90)
    )

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]

        # ----------- VERIFICATION ONLY -----------
        verified = False
        distance = None

        try:
            result = DeepFace.verify(
                img1_path=face_img,
                img2_path=REFERENCE_IMG,
                model_name=VERIFY_MODEL,
                distance_metric=VERIFY_METRIC,
                enforce_detection=False
            )
            verified = result.get("verified", False)
            distance = result.get("distance", None)
            print("Verify distance:", distance)
        except Exception as e:
            print("Verify error:", e)
            verified = False

        # ----------- DECISION -----------
        if verified and distance is not None and distance < VERIFY_THRESHOLD:
            label = "anukul chandra"
            color = (0, 255, 0)

            mark_attendance("anukul chandra")

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )

            cv2.imshow("Attendance System", frame)
            cv2.waitKey(1200)

            cap.release()
            cv2.destroyAllWindows()
            print("✅ Attendance complete. Camera closed.")
            exit()

        else:
            label = "Face not in database"
            color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
