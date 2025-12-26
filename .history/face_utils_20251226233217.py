import pickle
import numpy as np
from deepface import DeepFace

EMBEDDINGS_DIR = "embeddings"
THRESHOLD = 10

# Load embeddings once
db_embeddings = {}

for file in os.listdir(EMBEDDINGS_DIR):
    if file.endswith(".pkl"):
        name = file.replace(".pkl", "")
        with open(f"{EMBEDDINGS_DIR}/{file}", "rb") as f:
            db_embeddings[name] = pickle.load(f)

def recognize_face(image_path):
    rep = DeepFace.represent(
        img_path=image_path,
        model_name="Facenet",
        enforce_detection=False
    )

    live_embedding = np.array(rep[0]["embedding"])

    name_found = "Unknown"
    min_dist = float("inf")

    for name, embeddings in db_embeddings.items():
        for emb in embeddings:
            dist = np.linalg.norm(live_embedding - np.array(emb))
            if dist < min_dist:
                min_dist = dist
                name_found = name

    if min_dist < THRESHOLD:
        return name_found
    return "Unknown"
