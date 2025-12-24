import os
import pickle
from deepface import DeepFace

FACE_DB = "face_db"
EMBEDDINGS_DIR = "embeddings"

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

print("🔹 Generating face embeddings using DeepFace...")

for user in os.listdir(FACE_DB):
    user_path = os.path.join(FACE_DB, user)

    if not os.path.isdir(user_path):
        continue

    embeddings = []

    for img in os.listdir(user_path):
        img_path = os.path.join(user_path, img)

        try:
            result = DeepFace.represent(
                img_path=img_path,
                model_name="Facenet",
                enforce_detection=True
            )
            embeddings.append(result[0]["embedding"])
        except Exception as e:
            print(f"⚠️ Skipped {img_path}: {e}")

    if embeddings:
        with open(os.path.join(EMBEDDINGS_DIR, f"{user}.pkl"), "wb") as f:
            pickle.dump(embeddings, f)
        print(f"✅ Saved embeddings for {user}")
    else:
        print(f"❌ No embeddings for {user}")

print("🎉 Embedding generation completed!")
