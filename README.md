# Retina Guard

Diabetic retinopathy (DR) screening and progression monitoring from retinal fundus photographs.

Retina Guard grades a fundus image on the five-level APTOS severity scale with an EfficientNet-B3 classifier, explains each prediction with Grad-CAM heatmaps, and compares a baseline scan with a follow-up scan to flag possible progression. It ships with a Streamlit dashboard, a training script and an evaluation script.

> **Research and teaching prototype. Not a medical device.** Outputs must not be used for clinical decisions. See [Limitations](#limitations).

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Results](#results)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Running the dashboard](#running-the-dashboard)
- [Training](#training)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Limitations](#limitations)
- [References](#references)

## Features

- **Five-grade classification:** No DR, Mild, Moderate, Severe, Proliferative (APTOS 2019 scale).
- **Explainability:** Grad-CAM, Grad-CAM++ and Eigen-CAM overlays show which regions drove the prediction.
- **Retinal preprocessing:** fundus border crop, CLAHE contrast enhancement and optional Ben Graham local normalisation.
- **Progression analysis:** compares the Grad-CAM activation density of two scans and raises an alert when it rises by more than 10%.
- **Clinical-style output:** suggested follow-up interval and recommendations per grade, with a downloadable text report.
- **Bring your own weights:** upload a trained checkpoint in the sidebar, or train one with `train.py`.

## How it works

```
Fundus image
    |
    v
Crop border -> CLAHE (LAB colour space) -> Ben Graham normalisation -> 380x380 letterbox
    |
    v
EfficientNet-B3 (5-class head) -> grade + class probabilities
    |
    v
Grad-CAM on the last convolutional block -> heatmap + activation density
    |
    v
Prevention agent: relative change in density between two scans -> risk level + alert
```

**Activation density** is the fraction of the image where the normalised Grad-CAM activation exceeds 0.5. The prevention agent takes the relative change between a historical and a current scan:

| Relative change | Risk level | Alert |
|---|---|---|
| above +10% | High | Yes |
| above +5% | Moderate | No |
| otherwise | Low | No |

The follow-up interval is then looked up from the predicted grade and the risk level.

## Results

Evaluated on a held-out 20% stratified split of the APTOS 2019 training set (733 images, fixed seed, never seen during training). The split is reproduced exactly by `evaluate.py`.

| Metric | Value |
|---|---|
| Accuracy | 82.7% |
| Within one grade of the true grade | 95.4% |
| Quadratic weighted kappa | 0.886 |

Per-grade performance:

| Grade | Precision | Recall | F1 | Images |
|---|---|---|---|---|
| No DR | 0.986 | 0.961 | 0.973 | 361 |
| Mild | 0.558 | 0.649 | 0.600 | 74 |
| Moderate | 0.758 | 0.815 | 0.786 | 200 |
| Severe | 0.487 | 0.487 | 0.487 | 39 |
| Proliferative | 0.707 | 0.492 | 0.580 | 59 |

![Confusion matrix on the validation split](docs/confusion_matrix.png)

Collapsing the grades into "referable" (Moderate or worse) against "non-referable", the confusion matrix gives a sensitivity of 91.3% (272 of 298) and a specificity of 94.7% (412 of 435).

**Reading the results.** The model is strong on the large classes. Most errors are to a neighbouring grade, and the weakest grades are the rare ones: Severe and Proliferative recall are both below 50%. 25 of the 59 Proliferative images were predicted as Mild or Moderate, which is the clinically costly direction of error.

## Project structure

```
retina-guard/
├── app/
│   ├── main.py            Streamlit dashboard
│   ├── ui.py              Stylesheet and HTML components
│   ├── model.py           EfficientNet-B3, inference transform, prediction
│   ├── preprocessing.py   Crop, CLAHE, Ben Graham, resize
│   ├── gradcam.py         Grad-CAM / Grad-CAM++ / Eigen-CAM and activation density
│   └── prevention.py      Progression analysis and recommendations
├── tests/
│   └── test_pipeline.py   Tests for each module
├── docs/
│   └── confusion_matrix.png
├── .streamlit/
│   └── config.toml        Theme and toolbar settings
├── train.py               Fine-tune EfficientNet-B3 on APTOS 2019
├── evaluate.py            Metrics and confusion matrix on the validation split
├── requirements.txt
└── setup.py
```

## Installation

Python 3.10 to 3.12 is recommended.

```bash
git clone https://github.com/goelshubh-11/retina-guard.git
cd retina-guard

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Troubleshooting

- **`torch` cannot be found on an Intel Mac.** PyTorch stopped publishing Intel macOS builds after 2.2.2, which supports Python up to 3.12. Use Python 3.11 and pin the versions:
  ```bash
  pip install "numpy<2" "torch==2.2.2" "torchvision==0.17.2"
  pip install -r requirements.txt
  ```
- **OpenCV starts compiling and never finishes.** Install a prebuilt wheel instead, then install `grad-cam` without its dependencies:
  ```bash
  pip install --only-binary=:all: "opencv-python-headless==4.9.0.80"
  pip install --no-deps grad-cam
  pip install ttach
  ```
- **`numpy.core.multiarray failed to import`.** NumPy 2 is installed alongside a build that needs NumPy 1: run `pip install "numpy<2" --force-reinstall`.
- **`grad-cam` fails to build from PyPI.** Install it from source: `pip install git+https://github.com/jacobgil/pytorch-grad-cam.git`.

## Running the dashboard

```bash
python -m streamlit run app/main.py
```

Then open http://localhost:8501.

Model weights are **not included** in this repository. Without them the dashboard runs a baseline model (ImageNet backbone with an untrained classification head) and shows a notice, and its predictions are not meaningful. To get real predictions, either train a checkpoint (see below) or upload an existing one under **Model weights** in the sidebar. A `.pth` file containing either a plain state dict or a `{"model_state_dict": ...}` checkpoint is accepted. Checkpoints can be shared through GitHub Releases rather than committed to the repository.

**Single scan:** upload a fundus image to see the predicted grade, confidence, Grad-CAM overlay, class probabilities, recommendations and a downloadable report.

**Progression comparison:** upload a baseline and a follow-up scan to see both analyses side by side, the change in activation density, the risk level and any prevention alert.

## Training

Training uses the [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) dataset, which is not redistributed here. A GPU is strongly recommended: on a Kaggle T4, an epoch took about 11 to 12 minutes, largely because of CPU-side image preprocessing.

```bash
python train.py --data /path/to/aptos --epochs 8 --workers 4 --out retina_b3.pth
```

`--data` must contain `train.csv` and a `train_images/` folder. Use `--limit 64 --epochs 1` for a quick smoke test.

| Setting | Value |
|---|---|
| Backbone | EfficientNet-B3, ImageNet initialisation |
| Input | 380 x 380, same preprocessing as the dashboard |
| Optimiser | AdamW, learning rate 1e-4, weight decay 1e-2 |
| Schedule | Cosine annealing |
| Loss | Cross-entropy with square-root inverse-frequency class weights |
| Augmentation | Horizontal and vertical flips, rotation up to 30 degrees, brightness and contrast jitter |
| Precision | Mixed precision on CUDA |
| Split | 80/20 stratified, seed 42 |
| Model selection | Best validation quadratic weighted kappa |

The saved checkpoint contains only tensors and plain Python numbers, so it loads under PyTorch's default safe-loading mode.

## Evaluation

```bash
python evaluate.py --data /path/to/aptos --weights retina_b3.pth --workers 4
```

This reproduces the validation split used in training, prints accuracy, within-one-grade accuracy, quadratic weighted kappa and per-grade precision and recall, and writes `confusion_matrix.png` and `predictions.csv`.

## Tests

```bash
pytest tests/ -v
```

## Limitations

- **Validation, not independent testing.** The reported numbers come from a random split of a single dataset, and the best epoch was chosen on that same split, so they are somewhat optimistic. There has been no external or prospective validation.
- **Rare grades are weak.** Severe and Proliferative recall are below 50%, and Proliferative disease is often under-graded. A model with these error rates is not suitable for screening on its own.
- **Progression analysis is a heuristic.** Activation density is derived from where the network looks, not from a measured lesion count, and it has not been validated against clinical progression. It is also sensitive to image quality and field of view, so scans of the same eye should be comparable. A near-zero baseline density can inflate the relative change.
- **Explainability is not proof.** Grad-CAM shows where the network attended and may highlight regions that are not lesions.
- **Image quality.** The model was trained on APTOS photographs. Other cameras, compressed images or screenshots may behave differently.
- **Not a medical device.** Retina Guard has no regulatory clearance, and every output requires review by a qualified ophthalmologist.

## References

- APTOS 2019 Blindness Detection, Kaggle.
- M. Tan and Q. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019.
- R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017.
- A. Chattopadhay et al., "Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks", WACV 2018.
- S. Pizer et al., "Adaptive Histogram Equalization and Its Variations", 1987 (CLAHE).
- J. Gildenblat and contributors, `pytorch-grad-cam`.
