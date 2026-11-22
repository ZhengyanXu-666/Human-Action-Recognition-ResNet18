import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter

from dataset_fast import FastDataset
from model_fast import FastResNet18


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT = os.path.join(BASE_DIR, "data") # <-- this is the folder containing train/test videos
CSV_PATH = os.path.join(ROOT, "annotations", "train_set_labels.csv")


print("ROOT DIR:", ROOT)
print("CSV PATH:", CSV_PATH)


if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print("Using device:", DEVICE)


def evaluate_on_train(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    pred_counter = Counter()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            pred_counter.update(preds.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print("Train Accuracy (eval):", acc)
    print("Prediction distribution:", pred_counter)
    return acc


dataset = FastDataset(csv_path=CSV_PATH, root=ROOT)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = FastResNet18(num_classes=30).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 5
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    print(f"\nEpoch {epoch+1}/{EPOCHS} | Train Acc (batch): {acc:.2f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "fast_best.pth")
        print("  >>> Saved best model")

    evaluate_on_train(model, loader, DEVICE)

print("\nTraining complete. Best Acc:", best_acc)
