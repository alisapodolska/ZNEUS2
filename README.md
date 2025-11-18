# ZNEUS2
Project 2: Animal 10 multi-class classification

# Week 1 — Dataset Exploration & Preprocessing

This week focused on preparing the dataset for a 10-class animal image classification task. The work included exploratory analysis, cleaning, preprocessing, augmentation, and setting up data loaders for training.

---

## EDA

### Dataset Loading  

### Class Name Translation  
All class folders were renamed from Italian to English.

### Class Distribution  

### Image Size Inspection  
A sample of images was analyzed to check minimum, maximum, and average dimensions. This helped determine a consistent resize target.

### Corrupted Image Detection  
We found out that our dataset doesnt have corrupted images.

---

## Preprocessing

### Data Split  
The dataset was divided into **train**, **validation**, and **test** subsets using a standard ratio (70/20/10) while preserving class folders.

### Resize  
All images were resized to **224×224** to match typical CNN input requirements.

### Normalization  
Images were normalized using ImageNet mean and standard deviation:
- mean: `[0.485, 0.456, 0.406]`  
- std: `[0.229, 0.224, 0.225]`

---

## Augmentations

Applied only to the training set:

- Random horizontal flip  
- Random rotation  
- Color jitter  
- Random resized crop  
- Final resize and normalization  

### Augmentation Preview  
A batch of augmented images was visualized to confirm the transformations look natural.

---

## DataLoader Setup

PyTorch `DataLoader` objects were created for train, validation, and test sets.  
These loaders handle batching, shuffling (for training), and efficient image loading during training.


