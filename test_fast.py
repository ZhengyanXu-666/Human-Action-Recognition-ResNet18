# test_fast.py
import os
import torch
from torch.utils.data import DataLoader
from collections import Counter
import pandas as pd

from dataset_fast import FastDataset
from model_fast import FastResNet18

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   

ROOT = os.path.join(BASE_DIR, "data")  # dataset root
CSV_PATH = os.path.join(ROOT, "annotations", "train_set_labels.csv")
MODEL_PATH = os.path.join(BASE_DIR, "fast_best.pth")

print("ROOT:", ROOT)
print("CSV :", CSV_PATH)
print("MODEL:", MODEL_PATH)
# -------------------------------
# Device
# -------------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("Using:", DEVICE)

# -------------------------------
# Dataset & Loader
# -------------------------------
dataset = FastDataset(csv_path=CSV_PATH, root=ROOT)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# -------------------------------
# Model
# -------------------------------
model = FastResNet18(num_classes=30).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------------
# Evaluation
# -------------------------------
correct = 0
total = 0

video_ids = []
pred_labels = []
true_labels = []

with torch.no_grad():
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(imgs)
        _, preds = outputs.max(1)

        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        # record results
        video_ids.extend(dataset.df["video_id"].tolist()[len(video_ids): len(video_ids) + len(preds)])
        pred_labels.extend(preds.cpu().tolist())
        true_labels.extend(labels.cpu().tolist())

acc = correct / total
print("\nTest Accuracy:", acc)
print("Prediction distribution:", Counter(pred_labels))

# -------------------------------
# Save CSV results
# -------------------------------
# Shift labels back to 1–30
pred_shifted = [p + 1 for p in pred_labels]
true_shifted = [t + 1 for t in true_labels]

df = pd.DataFrame({
    "video_id": video_ids,
    "predicted_label": pred_shifted,
    "true_label": true_shifted
})

df.to_csv("test_results.csv", index=False)


print("\nSaved: test_results.csv")
