# import os
# import pickle
# from deepface import DeepFace

# FACE_DB = "face_db"
# EMBEDDINGS_DIR = "embeddings"

# os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# print("🔹 Generating face embeddings using DeepFace...")

# for user in os.listdir(FACE_DB):
#     user_path = os.path.join(FACE_DB, user)

#     if not os.path.isdir(user_path):
#         continue

#     embeddings = []

#     for img in os.listdir(user_path):
#         img_path = os.path.join(user_path, img)

#         try:
#             result = DeepFace.represent(
#                 img_path=img_path,
#                 model_name="Facenet",
#                 enforce_detection=True
#             )
#             embeddings.append(result[0]["embedding"])
#         except Exception as e:
#             print(f"⚠️ Skipped {img_path}: {e}")

#     if embeddings:
#         with open(os.path.join(EMBEDDINGS_DIR, f"{user}.pkl"), "wb") as f:
#             pickle.dump(embeddings, f)
#         print(f"✅ Saved embeddings for {user}")
#     else:
#         print(f"❌ No embeddings for {user}")

# print("🎉 Embedding generation completed!")

import os
import cv2
import pickle
import numpy as np
from deepface import DeepFace

# ---------------- CONFIG ----------------
FACE_DB_DIR = "face_db"        # enrolled images
EMBEDDINGS_DIR = "embeddings"  # output embeddings

MODEL_NAME = "Facenet"
DETECTOR = "retinaface"

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ---------------- PROCESS EACH USER ----------------
for person_name in os.listdir(FACE_DB_DIR):
    person_path = os.path.join(FACE_DB_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"\n🔄 Processing user: {person_name}")

    person_embeddings = []

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        try:
            # Generate embedding using SAME pipeline as recognition
            reps = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR,
                enforce_detection=True
            )

            embedding = np.array(reps[0]["embedding"])
            person_embeddings.append(embedding)

            print(f"  ✅ Embedded: {img_name}")

        except Exception as e:
            print(f"  ❌ Skipped {img_name}: {e}")

    # ---------------- SAVE EMBEDDINGS ----------------
    if len(person_embeddings) == 0:
        print(f"⚠️ No valid face found for {person_name}, skipping.")
        continue

    save_path = os.path.join(EMBEDDINGS_DIR, f"{person_name}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(person_embeddings, f)

    print(f"💾 Saved {len(person_embeddings)} embeddings → {save_path}")

print("\n🎉 All embeddings generated successfully!")
