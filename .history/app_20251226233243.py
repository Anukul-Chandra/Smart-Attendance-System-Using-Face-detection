from fastapi import FastAPI, UploadFile, File
import shutil
import os
from face_utils import recognize_face

app = FastAPI(title="Smart Attendance API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Smart Attendance API is running"}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    image_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    name = recognize_face(image_path)

    return {
        "recognized_person": name
    }
