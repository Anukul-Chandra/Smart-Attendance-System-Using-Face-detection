from fastapi import FastAPI, UploadFile, File
import shutil
import os

app = FastAPI(title="Smart Attendance API")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"message": "Smart Attendance API is running"}

@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "received",
        "filename": file.filename
    }
