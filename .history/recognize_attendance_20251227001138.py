import cv2
import os
import pickle
import numpy as np
from deepface import DeepFace

# ---------------- CONFIG ----------------
EMBEDDINGS_DIR = "embeddings"
MODEL_NAME = "Facenet"
DISTANCE_THRESHOLD = 0.55   # strict

# ---------------- LOAD EMBEDDINGS ----------------
db = {}

for file in os.listdir(EMBEDDINGS_DIR):
    if file.endswith(".pkl"):
        name = file.replace(".pkl", "")
        with open(os.path.join(EMBEDDINGS_DIR, file), "rb") as f:
            db[name] = pickle.load(f)

print("✅ Loaded users:", list(db.keys()))

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)
print("📷 Camera started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        reps = DeepFace.represent(
            img_path=frame,
            model_name=MODEL_NAME,
            enforce_detection=True
        )
        live_emb = np.array(reps[0]["embedding"])
    except:
        cv2.imshow("Attendance", frame)
        continue

    identity = "Unknown"
    min_dist = 999

    for name, embeddings in db.items():
        for emb in embeddings:
            dist = np.linalg.norm(live_emb - emb)
            if dist < min_dist:
                min_dist = dist
                identity = name

    print("DEBUG distance:", min_dist)

    if min_dist < DISTANCE_THRESHOLD:
        label = f"{identity} ✅"
        color = (0, 255, 0)
    else:
        label = "Unknown ❌"
        color = (0, 0, 255)

    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Attendance", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
