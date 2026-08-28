# Brain MRI Classification — Experiment Results

## Research Question

**Which image representation(along with an ML model) provides the most useful information for classifying brain MRI tumor types using classical machine learning?**

| Parameter                | Value                                                      |
| ------------------------ | ---------------------------------------------------------- |
| **Dataset**              | Brain Tumor MRI Dataset — Masoud Nickparvar                |
| **Total images**         | 7,023                                                      |
| **Classes**              | 4                                                          |
| **Classes**              | Glioma, Meningioma, No Tumor, Pituitary                    |
| **Image size**           | Resized to 128 × 128                                       |
| **Image representation** | Grayscale                                                  |
| **Raw-pixel dimensions** | 16,384 pixels/image                                        |
| **Dataset structure**    | Training / Testing                                         |
| **Random seed**          | 42                                                         |
| **Evaluation metrics**   | Accuracy, weighted Precision, weighted Recall, weighted F1 |
| **Preprocessing**        | Grayscale conversion → resize → flatten                    |
| **Model**                | Logistic Regression                                        |
| **Max iterations**       | 1,000                                                      |
---

## Experiment Overview

| Representation | Model               | Research Question                                                  |
| -------------- | ------------------- | ------------------------------------------------------------------ |
| Raw Pixels     | Logistic Regression | How far can raw visual information go?                             |
| HOG            | SVM                 | How useful is shape information?                                   |
| GLCM           | Random Forest       | How useful is texture information?                                 |
| HOG + GLCM     | SVM                 | Does combining shape and texture improve classification?           |
| PCA            | Logistic Regression | Can dimensionality reduction improve the representation?           |
| CNN            | CNN                 | How does learned representation compare with handcrafted features? |

---

# Results

| Representation | Model               | Accuracy | Precision | Recall | F1-Score | Training Time(Colab T4 GPU)              | Evaluation Time |
| -------------- | ------------------- | -------: | --------: | -----: | -------: | ---------------------------------------: | --------------: |
| Raw Pixels     | Logistic Regression |   0.8655 |    0.8639 | 0.8655 |   0.8641 |                                      58s |              2s |
| HOG            | SVM                 |   0.9705 |    0.9710 | 0.9705 |   0.9704 |   3m(2m for feature extration + 1m training) |             44s |
| GLCM           | Random Forest       |   0.8987 |    0.9003 | 0.8987 |   0.8976 | 7m2s(7m to Extract GLCM feat. 2s for training RF)|              4s |
| HOG + GLCM     | SVM                 |   0.9645 |    0.9653 | 0.9645 |   0.9643 | 13min(10m[Feature extraction], 3m[Standardize feat + train]) |  1m |
| PCA            | Logistic Regression |   0.8670 |    0.8649 | 0.8670 |   0.8656 | 8m(including PCA Feature extraction(6m)) |             ~0s |
| CNN            | CNN                 |   0.8345 |    0.8294 | 0.8345 |   0.8278 |                                      29m |              8m |
---

# Observations

## Raw Pixels

**Question:** How far can raw pixel information go?

**Result:** —

**Observation:** —

---

## PCA

**Question:** Can dimensionality reduction improve the representation?

**Result:** —

**Observation:** —

---

## HOG

**Question:** How useful is shape information?

**Result:** —

**Observation:** —

---

## GLCM

**Question:** How useful is texture information?

**Result:** —

**Observation:** —

---

## HOG + GLCM

**Question:** Does combining shape and texture improve classification?

**Result:** —

**Observation:** —

---

# Feature Representation Comparison

### Best-performing representation

**Representation:** —

**Accuracy:** —

### Worst-performing representation

**Representation:** —

**Accuracy:** —

### Key observation

—

---

# Classical ML vs CNN

| Approach     | Representation       | Model | Accuracy | F1-Score |
| ------------ | -------------------- | ----- | -------: | -------: |
| Classical ML | Handcrafted features | —     |        — |        — |
| CNN          | Learned features     | —     |        — |        — |

### Observation

—

---

# Final Inference

The experiments investigate whether **shape, texture, dimensionality reduction, or combinations of these representations** provide the most useful information for brain MRI tumor classification.

The final conclusion should be based on the experimental results rather than assuming that one representation will perform best beforehand.
