import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from model import EmbeddingNet  # from model.py

# === Configuration ===
VAL_DIR = "Task_B/val"
MODEL_PATH = "face_model.pth"
THRESHOLD = 0.6  # similarity threshold

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Image Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Cosine similarity ===
def cosine_similarity(x1, x2):
    return nn.functional.cosine_similarity(x1, x2)

# === Load model ===
model = EmbeddingNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === Evaluation ===
y_true = []
y_pred = []

print("🔍 Starting evaluation...")

for identity_folder in tqdm(os.listdir(VAL_DIR)):
    identity_path = os.path.join(VAL_DIR, identity_folder)
    if not os.path.isdir(identity_path):
        continue

    distortion_path = os.path.join(identity_path, "distortion")
    if not os.path.isdir(distortion_path):
        continue

    clean_images = [f for f in os.listdir(identity_path) if f.endswith(('.jpg', '.png'))]
    distorted_images = [f for f in os.listdir(distortion_path) if f.endswith(('.jpg', '.png'))]

    if len(clean_images) == 0:
        continue  # no reference image

    ref_img_path = os.path.join(identity_path, clean_images[0])
    ref_img = transform(Image.open(ref_img_path).convert("RGB")).unsqueeze(0).to(device)
    ref_embedding = model(ref_img)

    for d_img in distorted_images:
        test_img_path = os.path.join(distortion_path, d_img)
        test_img = transform(Image.open(test_img_path).convert("RGB")).unsqueeze(0).to(device)
        test_embedding = model(test_img)

        similarity = cosine_similarity(ref_embedding, test_embedding).item()
        predicted_label = 1 if similarity > THRESHOLD else 0

        # Simple label heuristic: distorted file name starts with identity folder name
        true_label = 1 if d_img.startswith(identity_folder) else 0

        y_true.append(true_label)
        y_pred.append(predicted_label)

# === Print Metrics ===
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n📊 Evaluation Results:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

