# ZNEUS2
# Project 2: Animal 10 Multi-Class Classification

**Team:** Alisa Podolska, Yulian Kisil  
**Task:** Image Classification (10 classes minimum) — Animals10 Dataset from Kaggle  
**Date:** December 01, 2025  

This project implements a multi-class image classification task for the Animals10 dataset, featuring 10 animal classes (cat, squirrel, cow, spider, sheep, elephant, horse, dog, butterfly, chicken). We adhered to the project requirements: exploratory data analysis (EDA), preprocessing/normalization/split/augmentations, configuration via JSON for dynamic models, experiment tracking with Weights & Biases (WandB), meaningful experiments with improvements (e.g., Dropout, Normalization layers like BatchNorm, Skip Connections in later models, LR scheduler), results with evaluation metrics (accuracy, loss, test eval), clear code with Markdown documentation/comments, and preparation for final presentation.

The project is divided into weeks: Week 1 (EDA & Preprocessing), Week 2 (Custom Dynamic CNN Experiments), Week 3 (Pretrained Models) Week 4 (more experiments, Grad-CAM, final metrics). All code is GPU-accelerated (CUDA). Evaluation metrics: Accuracy (primary for classification), CrossEntropyLoss; future: F1-score for imbalance.

---

## Week 1 — Dataset Exploration & Preprocessing

This week focused on loading, exploring, cleaning, splitting, and augmenting the Animals10 dataset (~26k images, unbalanced: spider=4821, elephant=1446). We performed comprehensive EDA to identify issues (e.g., variable sizes, color modes: RGB/RGBA/CMYK), ensured no corrupted images, and prepared data loaders for efficient training. This aligns with Consultation 1 minimal requirements: dataset download/examination, EDA example, metric selection (accuracy/loss), initial preprocessing/augmentations.

### Dataset Loading & Translation
- **Source:** Kaggle Animals10 (https://www.kaggle.com/datasets/alessiocorrado99/animals10).
- **Classes:** 10 animals; initial Italian names (e.g., "cane" → "dog") were translated/renamed via mapping dictionary (one-time script).
- **Total Images:** ~26,000 (unbalanced distribution visualized via bar plot).

**Class Distribution:**
| Class     | Count  |
|-----------|--------|
| cat       | 1668  |
| squirrel  | 1862  |
| cow       | 1866  |
| spider    | 4821  |
| sheep     | 1820  |
| elephant  | 1446  |
| horse     | 2623  |
| dog       | 4863  |
| butterfly | 2112  |
| chicken   | 3098  |

- **Insight:** Imbalance noted, spider/dog dominate.

### Visual Sanity Check
- Random sample (1 image per class) visualized via matplotlib grid (figsize=24x8).
- Confirmed diverse animal poses/views; no obvious labeling errors.

### Image Size & Format Inspection
- Sampled 200 images/class: Average size ~329x253 px; Min: 60x57; Max: 4928x3264.
- Boxplot of widths/heights: High variance (outliers from ultra-HD images).
- Channels: {'RGB', 'RGBA', 'CMYK'} — transforms handle conversion to RGB.
- **Insight:** Resize to 224x224 (ImageNet standard) to standardize.

### Corrupted Image Detection
- Verified all images with PIL `verify()`: **0 corrupted** (bad=[]). Dataset clean.

### Data Split
- Ratio: **Train 70%**, **Val 20%**, **Test 10%** (stratified via train_test_split, random_state=42).
- Folders created: `dataset/train/val/test/{class}`; files copied.
- Preserves class balance in splits.

### Resize & Normalization
- **Target:** 224x224.
- **Transforms (Val/Test):** Resize → ToTensor → Normalize (ImageNet stats: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
- **Post-Transform Check:** Boxplot confirmed uniform 224x224; sample visualization (permute for RGB display).

### Augmentations (Train-Only)
To combat imbalance/overfitting: Compose chain applied only to train.
- RandomHorizontalFlip (p=0.5): Mirrors for symmetry.
- RandomRotation(15°): Orientation variance.
- ColorJitter(brightness/contrast/saturation=0.2): Lighting robustness.
- RandomResizedCrop(224, scale=0.8-1.0): Scale/position variation.
- Final: Resize → ToTensor → Normalize.
- **Preview:** Visualized augmented samples per class (PIL conversion); transformations natural (no artifacts).
- **Rationale:** Augmentations increase effective dataset size ~2-3x, improve generalization (e.g., for variable lighting in wild animal photos).

### DataLoader Setup
- PyTorch ImageFolder + DataLoader: Batch=32, shuffle=True (train), num_workers=0 (default).
- Post-aug check: Dimensions uniform (3x224x224 tensors); class indices mapped correctly.

**Week 1 Outcomes:** Clean, balanced pipeline ready for training. Total prep time: ~10 min on CPU. Aligns with requirements: Data analysis (EDA/visuals/stats), preprocessing/normalization/split, augmentations, config (transforms as proto-JSON).

---

## Week 2 — Custom Dynamic CNN Experiments

Implemented a **fully dynamic CNN** (`ConfigurableAnimalModel`) built from JSON configs: conv-blocks (variable depth/filters), optional BatchNorm, dynamic FC-layers (list of hidden sizes), dropout, auto-flatten. Supports in_channels/num_classes from config; static `from_config()` method. Trained 3 variants (v1 basic, v2 deeper, v3 BN+SGD+scheduler) for 15-25 epochs (Adam/SGD, lr=1e-4 to 0.01). Metrics: Train/Val/Test Acc, Loss; logged to WandB. Improvements: Dropout=0.5/0.3, Normalization (BN in v3), LR scheduler (StepLR in v3).

**Results Summary (from WandB runs):**
| Variant | Blocks | Optimizer/LR | Epochs | Val Acc (Final) | Test Acc | Key Improvement |
|---------|--------|--------------|--------|-----------------|----------|-----------------|
| v1     | 4     | Adam/1e-4   | 15    | 0.6063         | 0.60    | Baseline (1 FC layer) |
| v2     | 5     | Adam/5e-5   | 20    | 0.79           | 0.73    | Deeper conv (better features) |
| v3     | 5     | SGD/0.01 + Scheduler | 25 | 0.87           | 0.86    | BN + adaptive LR (stability) |

- **Dynamic Features:** JSON controls all (e.g., fc_layers=[{"in":null,"out":1024},{"in":1024,"out":512}] for multi-hidden in v3). No code changes needed.
- **Tracking:** WandB logs epoch-wise acc/loss/LR; configs auto-saved for reproducibility.
- **Insights:** v3 best (BN reduces overfitting; scheduler stabilizes late epochs). Total params: v1~5M, v3~10M.

Aligns with requirements: Configuration (JSON), Experiments (progressive improvements: depth, normalization, scheduler), Tracking (WandB), Results/Metrics (acc/loss/test), Clear code/comments.

---

## Week 3 — Pretrained Model Baselines

Benchmarked 6 pretrained ImageNet models (Torchvision) as baselines: LeNet (simple), AlexNet/VGG16 (classic), Inception-v1 (multi-branch), ResNet-50 (residual), DenseNet-121 (dense), MobileNet-v2 (lightweight). All fine-tuned (replace final FC to 10 classes, lr=1e-4 Adam, 15 epochs). Special handling: Inception aux-loss (weighted 0.3). Metrics logged to WandB; test eval added.

**Results Summary pre-trained versions (from WandB runs):**
| Model       | Params (M) | Val Acc (Final) | Test Acc | Notes |
|-------------|------------|-----------------|----------|-------|
| LeNet      | 0.06      | 0.6063         | 0.60    | Baseline; simple conv (adapted for 224x224). |
| AlexNet    | 60        | 0.96           | 0.97   | Fast convergence (~0.83→0.95); good transfer. |
| VGG16      | 138       | 0.97           | 0.97    | Deep uniform conv; stable but compute-heavy. |
| Inception-v1 | 7       | 0.99           | 0.99    | Aux classifiers aid early learning; multi-scale. |
| ResNet-50  | 25        | 0.98           | 0.98    | Residuals prevent degradation. |
| DenseNet-121 | 8       | 0.99           | 0.99    | Feature reuse; efficient for detail-rich animals. |
| MobileNet-v2 | 3.5     | 0.93           | 0.92    | Lightweight; suitable for edge deployment. |
| ResNet-50(not pre-t.) | 25        | 0.74           | 0.70    | Residuals prevent degradation. |

- **Training:** Reused train_epoch/eval_epoch; Inception custom (tuple outputs).
- **Insights:** Pretrained > custom (90%+ vs 70%); ResNet/DenseNet top (skip/dense connections = improvements). Overfitting minimal (augmentations help).
- **WandB:** Separate runs; config logs model metrics.

---

## Week 4 — Final changes
## Summary

During this week, we defined and partially executed the following plan:
- Run more experiments  
- Test non-pretrained versions of existing architectures (AlexNet, VGG16, etc.)  
- Implement Grad-CAM as a bonus task  
- Evaluate the best-performing model  
- Compute final metrics and confusion matrix  
- Clean the codebase and add structured markdown documentation  

### Implemented Experiments and Results

- A **non-pretrained Inception model** was trained, but it showed **negative performance results** and did not reach a satisfactory accuracy level.  
- A **non-pretrained ResNet model** was also trained and achieved **positive results**, with an approximate val accuracy of **74%**. However, this is still **worse than the performance of our V3 custom model**.

### Model Evaluation Improvements

- An **improved evaluation pipeline** was implemented for the **V3 model**, including:
  - Confusion Matrix  
  - ROC Curve  
  - Additional classification metrics  

### Grad-CAM Implementation

- **Grad-CAM visualization** was successfully implemented.
- Different Grad-CAM cases were analyzed, including both correct and incorrect model attention examples.
- Detailed interpretations of these cases are provided directly in the notebook.

### Notebook Refinement

- The entire notebook was **better structured and reorganized**.
- Additional **comments and markdown sections** were added for easier readability.
- The codebase was **cleaned and optimized**, and unnecessary outputs were removed.

Overall, this week focused on deeper experimentation, model interpretability, and improving the clarity and structure of the entire project. 

