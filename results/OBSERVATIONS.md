# Brain MRI Classification — Experiment Results

## Research Question

**Which image representation provides the most useful information for classifying brain MRI tumor types using classical machine learning?**

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

| Representation | Model               | Accuracy | Precision | Recall | F1-Score |
| -------------- | ------------------- | -------: | --------: | -----: | -------: |
| Raw Pixels     | Logistic Regression |   0.8655 |    0.8639 | 0.8655 |   0.8641 |
| HOG            | SVM                 |        — |         — |      — |        — |
| GLCM           | Random Forest       |        — |         — |      — |        — |
| HOG + GLCM     | SVM                 |        — |         — |      — |        — |
| PCA            | Logistic Regression |        — |         — |      — |        — |
| CNN            | CNN                 |        — |         — |      — |        — |
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
