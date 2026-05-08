# 🫁 Tuberculosis Detection using EfficientNetB0 with TTA

Deep learning project for detecting Tuberculosis from chest X-ray images using transfer learning and EfficientNetB0.

---

## 📌 Project Overview

This project focuses on binary classification of chest X-ray images:

- **0 → Normal**
- **1 → Tuberculosis**

The goal was to build a model that generalizes well on unseen data and performs strongly on the competition leaderboard.

---

## 🧠 Model Architecture

### 🔹 Base Model
- EfficientNetB0 (pretrained on ImageNet)
- `include_top=False`

### 🔹 Custom Head
- GlobalAveragePooling2D
- Dense (256 units, ReLU)
- Dropout (0.4)
- Dense (1 unit, Sigmoid)

---

## ⚙️ Techniques Used

- Transfer Learning
- Fine-tuning
- Dropout Regularization
- Test Time Augmentation (TTA)
- Ensembling
- EarlyStopping
- EfficientNet preprocessing (`preprocess_input`)

---

## 🔄 Data Augmentation

Training augmentations:
- Horizontal Flip
- Rotation
- Zoom

TTA:
- Horizontal Flip

---

## 📈 Results

| Setup | Public AUC | Private AUC |
|------|------------|-------------|
| Baseline Model | ~0.88 | ~0.88 |
| Improved Model | ~0.91 | ~0.92 |
| Final Model + TTA | ~0.959 | **~0.9628** |

---

## 🚀 Key Learnings

- Regularization significantly improved generalization
- Public leaderboard score alone can be misleading
- TTA improved prediction stability
- Proper preprocessing is critical in transfer learning workflows

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib

---

## 📂 Project Structure

```text
├── notebook.ipynb
├── README.md
├── submission.csv
└── model_checkpoints/
```

---

## ✨ Future Improvements

- Explore Grad-CAM visualization
- Experiment with EfficientNetB1/B2
- Try advanced ensembling strategies

---

## 🙌 Acknowledgements

Dataset and evaluation platform provided through a deep learning competition challenge.
