# dataset_fast.py
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class FastDataset(Dataset):
    """
    Fast dataset for action recognition:
    - CSV format: video_id, class_name, label
    - Uses the 15th frame from each video
    """

    def __init__(self, csv_path, root):
        self.root = root

        self.df = pd.read_csv(csv_path, header=None)
        self.df.columns = ["video_id", "class_name", "label"]
        self.df["label"] = self.df["label"].astype(int) - 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_id = row["video_id"]
        label = row["label"]

        video_path = os.path.join(self.root, "train_set", video_id + ".avi")

        cap = cv2.VideoCapture(video_path)

        target_frame_idx = 15  
        frame = None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            frame = 255 * np.ones((224, 224, 3), dtype=np.uint8)
        else:
            if target_frame_idx >= total_frames:
                target_frame_idx = total_frames - 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ok, frame = cap.read()

            if not ok:
                frame = 255 * np.ones((224, 224, 3), dtype=np.uint8)

        cap.release()

        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
        label = torch.tensor(label).long()

        return img, label
