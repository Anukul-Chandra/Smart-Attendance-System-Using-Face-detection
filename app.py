from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import pickle
import tempfile
from deepface import DeepFace

# ---------------- CONFIG ----------------
EMBEDDINGS_DIR = "embeddings"
MODEL_NAME = "Facenet"
DISTANCE_THRESHOLD = 0.55   # strict threshold

app = FastAPI(
    title="Smart Attendance API",
    version="1.0.0"
)

# ---------------- LOAD EMBEDDINGS ----------------
db = {}

for file in os.listdir(EMBEDDINGS_DIR):
    if file.endswith(".pkl"):
        name = file.replace(".pkl", "")
        with open(os.path.join(EMBEDDINGS_DIR, file), "rb") as f:
            db[name] = pickle.load(f)

print("✅ Loaded users:", list(db.keys()))

# ---------------- ROOT ----------------
@app.get("/")
def root():
    return {"message": "Smart Attendance API is running"}

# ---------------- RECOGNIZE API ----------------
@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    try:
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            contents = await file.read()
            tmp.write(contents)
            img_path = tmp.name

        img = cv2.imread(img_path)
        if img is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid image"}
            )

        # Face embedding
        reps = DeepFace.represent(
            img_path=img,
            model_name=MODEL_NAME,
            enforce_detection=True
        )

        live_emb = np.array(reps[0]["embedding"])

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
            return {
                "recognized_person": identity,
                "distance": round(min_dist, 4),
                "status": "matched"
            }
        else:
            return {
                "recognized_person": "Unknown",
                "distance": round(min_dist, 4),
                "status": "not_in_database"
            }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
