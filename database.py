from deepface import DeepFace
import os
import pickle

DB_PATH = "dataset"
MODEL = "ArcFace"

database = []

for person in os.listdir(DB_PATH):
    person_path = os.path.join(DB_PATH, person)
    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        try:
            emb = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL,
                detector_backend="mtcnn",
                enforce_detection=True
            )[0]["embedding"]

            database.append({
                "name": person,
                "embedding": emb
            })

            print(f"✅ {img_path}")

        except:
            print(f"⚠️ Пропущено: {img_path}")

with open("face_db.pkl", "wb") as f:
    pickle.dump(database, f)

print(f"🎉 База сохранена: {len(database)} лиц")
