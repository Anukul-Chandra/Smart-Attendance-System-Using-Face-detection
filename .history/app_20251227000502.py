from fastapi import FastAPI, UploadFile, File
import shutil
import os
import uuid

from face_utils import recognize_face

app = FastAPI(title="Smart Attendance API")

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- ROOT ----------------
@app.get("/")
def root():
    return {"message": "Smart Attendance API is running"}

# ---------------- RECOGNIZE ----------------
@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    # save uploaded image
    file_ext = file.filename.split(".")[-1]
    temp_filename = f"{uuid.uuid4()}.{file_ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # recognize
    result = recognize_face(temp_path)

    # cleanup
    os.remove(temp_path)

    return result
