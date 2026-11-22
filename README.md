# Human Action Recognition with ResNet18

A PyTorch-based computer vision project for recognizing human actions in industrial human–robot interaction videos. 

The implementation provided here uses a pretrained **ResNet18** image-classification model. For each video, the dataset loader extracts one frame (frame index 15, or the final available frame if the video is shorter), resizes it to 224 × 224 pixels, and uses it as the model input.

> **Important:** The current `test_fast.py` script evaluates the labelled data in `data/train_set/`. It does **not** run inference on the separate, unlabelled `data/test_set/` videos. Therefore, the accuracy printed by this script must not be described as unseen-test-set accuracy.

## 1.Project structure

Put the four Python scripts and the README in the **project root**. The `data` directory must be at the same level as those scripts.

```text
SAP-Human-Action-Recognition/
├── README.md
├── train_fast.py
├── test_fast.py
├── dataset_fast.py
├── model_fast.py
├── fast_best.pth              # Created by training
├── test_results.csv           # Created by test_fast.py
└── data/
    ├── annotations/
    │   └── train_set_labels.csv
    ├── train_set/
    │   ├── CID01_SID01_VID01.avi
    │   ├── CID01_SID01_VID02.avi
    │   └── ...                # Training videos
    └── test_set/
        ├── testvideo001.avi
        ├── testvideo002.avi
        └── ...                # Separate test videos
```

Do not put the scripts inside another `code/` directory unless you also update the paths in the scripts. The scripts determine their base directory from their own file location and expect `data/` to be alongside them.

### Dataset files expected by the current code

| Item | Expected location | Used by current scripts |
|---|---|---|
| Training labels | `data/annotations/train_set_labels.csv` | Yes |
| Labelled training videos | `data/train_set/` | Yes |
| Separate test videos | `data/test_set/` | No, not by the current `test_fast.py` |
| Trained model weights | `fast_best.pth` in project root | Loaded by `test_fast.py` |

The CSV is read without a header (`header=None`) and is expected to contain three columns in this order:

```text
video_id,class_name,label
```

The `video_id` values are used to locate videos named `<video_id>.avi` in `data/train_set/`. Labels are converted from 1-based values to 0-based class indices internally.

**Dataset access and permissions:** The original coursework dataset may be subject to university or dataset-owner access restrictions. Do not upload the video files, annotations, or other restricted dataset material to a public GitHub repository unless you have explicit permission. Users should obtain the dataset through an authorised source and place it in the directory structure above.

## 2. Requirements

- Python 3.9 or later is recommended.
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pandas

The code automatically selects an available device in this order:
1. CUDA, if available
2. Apple Silicon MPS, if available and built
3. CPU otherwise

## 3. Set up the environment

Open Terminal and move into the project root directory:

```bash
cd /path/to/SAP-Human-Action-Recognition
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Upgrade pip and install the dependencies:

```bash
python -m pip install --upgrade pip
pip install torch torchvision pandas numpy opencv-python
```

For installation instructions specific to your operating system or accelerator, use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/). The appropriate PyTorch command can differ depending on macOS, Windows, Linux, CUDA, and hardware.

## 4. Train the model

Before training, check that:
- `train_fast.py`, `dataset_fast.py`, and `model_fast.py` are in the project root.
- `data/annotations/train_set_labels.csv` exists.
- The corresponding `.avi` files are in `data/train_set/`.

Run:

```bash
python train_fast.py
```

The current training script:
- Loads the labelled training dataset.
- Uses a ResNet18 model initialized with ImageNet pretrained weights.
- Replaces the final fully connected layer with a 30-class output layer.
- Uses cross-entropy loss and the Adam optimizer with a learning rate of `1e-4`.
- Trains for 5 epochs with a batch size of 32.
- Saves the model state when the script records a new best training accuracy.

The checkpoint is saved as:

```text
fast_best.pth
```

Run the command from the project root so that the checkpoint is created in the expected location.

## 5. Run the current evaluation script

After training has created `fast_best.pth`, run:

```bash
python test_fast.py
```

The script loads `fast_best.pth`, creates a `FastDataset` from `data/annotations/train_set_labels.csv`, and reads the corresponding videos from `data/train_set/`. It then compares predictions with those same CSV labels.

The script writes:

```text
test_results.csv
```

The output CSV contains:

```text
video_id,predicted_label,true_label
```

The labels in this output are shifted back to the range 1–30.

> **Evaluation limitation:** Despite the filename `test_fast.py`, this script currently evaluates the labelled training set. The `FastDataset` class constructs every video path using the `train_set` directory, and `test_fast.py` passes the training-label CSV. The separate `data/test_set/` directory is not used. A separate inference/evaluation implementation would be needed to generate predictions for unlabelled test videos.

## 6. Model and data pipeline

### Model

`model_fast.py` defines `FastResNet18`:
- Backbone: Torchvision ResNet18
- Initial weights: `ResNet18_Weights.IMAGENET1K_V1`
- Final layer: 512 input features mapped to 30 output classes

### Frame extraction

`dataset_fast.py` defines `FastDataset`:
1. Reads the CSV columns `video_id`, `class_name`, and `label`.
2. Opens the matching `.avi` file under `data/train_set/`.
3. Selects frame index 15. If the video has fewer frames, it selects the final frame.
4. Resizes the frame to 224 × 224.
5. Converts BGR to RGB.
6. Converts the image to a PyTorch tensor and scales pixel values to the range 0–1.

This is a single-frame baseline; it does not process the full temporal sequence of a video.

## 7. Reproducibility notes and current limitations

- The scripts use a fixed frame index (15) rather than sampling multiple frames.
- The model predicts 30 classes.
- The training script reports accuracy on the training data; this is not a validation-set result.
- The current evaluation script also uses the labelled training data, not the separate test videos.
- No independent validation split or unseen-test evaluation is implemented in the supplied scripts.
- Results can vary with software versions, hardware, and training conditions. The supplied scripts do not set random seeds.

