# 🔬 Retina Guard

> AI-Powered Diabetic Retinopathy Detection & Progression Monitoring  
> EfficientNet-B3 · Grad-CAM · CLAHE · Streamlit

---

## Overview

**Retina Guard** is a full end-to-end medical AI platform for detecting and tracking
Diabetic Retinopathy (DR) from fundus photographs. It is built for clinical education
and research purposes.

### Detection-to-Prevention Pipeline

```
Upload ──► CLAHE + Ben Graham ──► EfficientNet-B3 ──► Grad-CAM ──► Prevention Agent ──► Report
```

| Stage | Technology | Detail |
|-------|-----------|--------|
| **Preprocessing** | OpenCV CLAHE | Contrast-Limited Adaptive Histogram Equalisation in LAB space to lift microaneurysm visibility |
| **Local normalisation** | Ben Graham method | Subtract Gaussian-blurred image to remove illumination artefacts |
| **Classification** | EfficientNet-B3 (PyTorch) | 5-class DR severity grading (APTOS 2019 scale) |
| **Explainability** | Grad-CAM / Grad-CAM++ / Eigen-CAM | Heatmaps highlighting lesion regions for doctor verification |
| **Progression analysis** | Prevention Agent | Delta-change on Grad-CAM activation density across two scans |
| **Alerting** | Rules engine | 🔴 Alert if lesion density increases > 10 % relative |

---

## DR Severity Classes (APTOS 2019)

| Class | Label | Action |
|-------|-------|--------|
| 0 | No DR | Annual screening |
| 1 | Mild DR | 9-12 month review |
| 2 | Moderate DR | Ophthalmology within 3 months |
| 3 | Severe DR | Urgent referral within 1 month |
| 4 | Proliferative DR | Immediate referral, anti-VEGF / laser |

---

## Project Structure

```
retina_guard/
├── app/
│   ├── __init__.py
│   ├── main.py            ← Streamlit GUI (run this)
│   ├── model.py           ← EfficientNet-B3 architecture + inference
│   ├── gradcam.py         ← Grad-CAM / Grad-CAM++ / Eigen-CAM
│   ├── preprocessing.py   ← CLAHE + Ben Graham + fundus crop
│   └── prevention.py      ← Progression agent + clinical report
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py   ← 20+ pytest tests covering all modules
├── requirements.txt
├── setup.py
└── README.md
```

---

## Setup & Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app/main.py
```

---

## Using Custom Model Weights

Retina Guard ships in **demo mode** using ImageNet-pretrained EfficientNet-B3 weights.
To use fine-tuned DR weights (e.g. trained on APTOS 2019):

1. Fine-tune EfficientNet-B3 for 5-class classification (see training section below).
2. Save: `torch.save(model.state_dict(), "retina_b3.pth")`
3. In the sidebar → upload your `.pth` file → click **Reload model**.

### Quick training snippet (APTOS dataset)

```python
import torch, torchvision
from app.model import build_model, NUM_CLASSES

model     = build_model(pretrained=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

for epoch in range(20):
    for imgs, labels in train_loader:
        imgs, labels = imgs.cuda(), labels.cuda()
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
    scheduler.step()

torch.save(model.state_dict(), "retina_b3_aptos.pth")
```

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

Expected test classes:
- `TestPreprocessing` — CLAHE, crop, resize, dtype checks
- `TestModel`         — architecture, classifier head, predict API
- `TestGradCAM`       — overlay shape, value ranges, both CAM methods
- `TestPreventionAgent` — delta-change, risk tiers, recommendations, alerts

---

## GUI Features

### Single Scan Mode
- Upload one fundus photograph
- See CLAHE-enhanced + Ben Graham preprocessed image
- Grad-CAM overlay highlighting lesion hotspots
- 5-class probability chart
- Clinical recommendations + next review date

### Progression Comparison Mode
- Upload baseline + current scan
- Side-by-side Grad-CAM comparison
- Δ-density bar chart
- 🔴 Prevention Alert if relative spread > 10 %
- Structured clinical report

---

## Ethical Disclaimer

> **Retina Guard is a research and educational tool.  
> It is NOT a certified medical device and must NOT be used as a  
> substitute for professional clinical diagnosis.**  
> All outputs must be reviewed by a qualified ophthalmologist.

---

## References

- APTOS 2019 Blindness Detection — Kaggle
- EfficientNet: Rethinking Model Scaling for CNNs (Tan & Le, 2019)
- Grad-CAM: Visual Explanations from Deep Networks (Selvaraju et al., 2017)
- Contrast Limited Adaptive Histogram Equalization (Pizer et al., 1987)
- Ben Graham, Kaggle DR Detection, 2015
