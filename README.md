# Dental-Image-Classification-CNN-nad-ResNet50
# 🦷 Dental Image Classification with PyTorch

A deep learning project for multi-class classification of dental photographs, using both a custom CNN and transfer learning with ResNet50.  
Designed with **orientation safety**, **class imbalance handling**, and **explainability** in mind — suitable for medical imaging pipelines.

---

## 📌 Project Overview
- **Goal:** Classify intraoral dental images into 8 anatomical views:
  - Lower Front, Lower Occlusal, Upper Right, Lower Right, Upper Left, Upper Occlusal, Upper Front, Lower Left
- **Challenges addressed:**
  - Class imbalance (weighted loss / sampler)
  - No horizontal/vertical flips to preserve anatomical correctness
  - Grad-CAM explainability for clinical trust

---

## ⚙️ Workflow
1. **Preprocessing**
   - DataFrame-based file management (paths + labels)
   - Stratified train/val/test split (70/15/15)
   - Class balancing strategies
   - Orientation-safe augmentations (rotations, color jitter, crop)

2. **Modeling**
   - **SmallCNN**: ~1.2M params, trained from scratch
   - **ResNet50**: fine-tuned with ImageNet weights
   - Mixed-precision training, early stopping, cosine LR schedule

3. **Evaluation**
   - Accuracy, per-class metrics, confusion matrices
   - Grad-CAM overlays for model interpretability

---

## Results (Example)
| Model              | Data Used | Val Accuracy | Test Accuracy |
|--------------------|-----------|--------------|---------------|
| SmallCNN (subset)  | 50%       | 0.92         | 0.90          |
| SmallCNN (full)    | 100%      | 0.96         | 0.95          |
| ResNet50 (full)    | 100%      | 0.98         | 0.97          |

> *Metrics will vary depending on training runs and hardware.*

---

