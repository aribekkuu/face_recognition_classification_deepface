# from deepface import DeepFace
# import cv2
# import os

# DB_PATH = "dataset"
# TEST_IMAGE = "IMG_4426.JPG"   # фото для проверки

# if not os.path.exists(TEST_IMAGE):
#     raise FileNotFoundError("test.jpg не найден")

# print("🔍 Searching face")

# result = DeepFace.find(
#     img_path=TEST_IMAGE,
#     db_path=DB_PATH,
#     model_name="ArcFace",
#     detector_backend="mtcnn",
#     enforce_detection=True
# )

# if len(result) == 0 or result[0].empty:
#     print("❌ Face does not found in database")
# else:
#     identity = result[0].iloc[0]["identity"]
#     name = identity.split("/")[-2]
#     distance = result[0].iloc[0]["distance"]

#     print(f"✅ Found person: {name}")
#     print(f"📏 Distance: {distance:.4f}")


# from deepface import DeepFace
# import os
# import numpy as np
# from scipy.spatial.distance import cosine

# DB_PATH = "dataset"
# TEST_IMAGE = "IMG_4426.JPG"
# MODEL = "ArcFace"
# THRESHOLD = 0.45

# print("📦 Uploading face database...")

# database = []

# for person in os.listdir(DB_PATH):
#     person_path = os.path.join(DB_PATH, person)
#     if not os.path.isdir(person_path):
#         continue

#     for img_name in os.listdir(person_path):
#         img_path = os.path.join(person_path, img_name)

#         try:
#             emb = DeepFace.represent(
#                 img_path=img_path,
#                 model_name=MODEL,
#                 detector_backend="mtcnn",
#                 enforce_detection=True
#             )[0]["embedding"]

#             database.append({
#                 "name": person,
#                 "embedding": emb
#             })

#         except Exception as e:
#             print(f"⚠️ Пропущено: {img_path}")

# print(f"✅ In database: {len(database)} лиц")

# print("🔍 Checking test image")

# test_emb = DeepFace.represent(
#     img_path=TEST_IMAGE,
#     model_name=MODEL,
#     detector_backend="mtcnn",
#     enforce_detection=True
# )[0]["embedding"]

# best_match = None
# best_distance = 1.0

# for item in database:
#     dist = cosine(test_emb, item["embedding"])
#     if dist < best_distance:
#         best_distance = dist
#         best_match = item["name"]

# print(f"📏 Min distance: {best_distance:.4f}")

# if best_distance < THRESHOLD:
#     print(f"✅ Found person: {best_match}")
# else:
#     print("❌ Unknown person")


from deepface import DeepFace
import pickle
from scipy.spatial.distance import cosine

MODEL = "ArcFace"
THRESHOLD = 0.45

with open("face_db.pkl", "rb") as f:
    database = pickle.load(f)

test_emb = DeepFace.represent(
    img_path="IMG_4426.JPG",
    model_name=MODEL,
    detector_backend="mtcnn",
    enforce_detection=True
)[0]["embedding"]

best_match = None
best_distance = 1.0

for item in database:
    dist = cosine(test_emb, item["embedding"])
    if dist < best_distance:
        best_distance = dist
        best_match = item["name"]

print(f"📏 Distance: {best_distance:.4f}")

if best_distance < THRESHOLD:
    print(f"✅ Found person: {best_match}")
else:
    print("❌ Unknown person")

