# BrainMRI - ML
This project explores the implementation of ClassicalML for Differentiating Brain Tumor from MRI Scans. The types of tumor i'm training the model for are glioma, meningioma, pituitary & normal brain without any tumor. There are 7023 images in the dataset. glioma - 300, meningioma - 306, pituitary - 300, no tumor - 405.

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

**Result** - 86.55% accuracy, 86.41% F1-score

**Observation** — Raw grayscale pixel values provided a reasonably strong baseline, achieving 86.55% accuracy. This indicates that the original pixel intensities contain useful information for distinguishing the four tumor classes. However, treating every pixel as an independent feature does not explicitly capture important structures such as edges, contours, or local texture patterns. The result therefore provides a useful baseline against which the more structured representations can be compared.

---

## PCA

**Question:** Can dimensionality reduction improve the representation?

**Result** — 86.70% accuracy, 86.56% F1-score

**Observation** — PCA produced only a **small improvement** over the raw pixel baseline, increasing accuracy from 86.55% to 86.70%. This suggests that dimensionality reduction alone did **not substantially improve the information available to the logistic regression model.** While PCA can compress correlated pixel information and reduce dimensionality, it does not specifically encode discriminative visual properties such as shape or texture.

---

## HOG

**Question:** How useful is shape information?

**Result:** — 97.05% accuracy, 97.04% F1-score

**Observation** — HOG produced the **strongest result** among all tested representations, achieving 97.05% accuracy and a 97.04% F1-score with an **SVM.** This represents a substantial improvement over the raw-pixel baseline. HOG describes local intensity gradients and captures structural information such as edges, contours, and shapes. The strong performance suggests that shape and structural information is highly discriminative for this dataset.

---

## GLCM

**Question:** How useful is texture information?

**Result** — 89.87% accuracy, 89.76% F1-score

**Observation** — GLCM-based texture features achieved 89.87% accuracy, outperforming the raw-pixel baseline but remaining considerably below HOG. This indicates that texture information contains useful discriminatory information, but texture alone was less effective than the shape based HOG representation for this classification task. The relatively high feature extraction time also shows that GLCM was computationally more expensive in this implementation.

---

## HOG + GLCM

**Question:** Does combining shape and texture improve classification?

**Result** — 96.45% accuracy, 96.43% F1-scor

**Observation** — Combining HOG and GLCM produced 96.45% accuracy, which was **slightly lower than HOG alone** at 97.05%. Therefore, adding the GLCM texture features **did not improve performance** in this experiment. This suggests that either the **additional texture** information provided **limited complementary information,** or that the larger combined feature space **introduced redundancy/noise that affected the SVM.** The combined representation also required substantially **more feature extraction time.**

---

# Feature Representation Comparison

### Best-performing representation

**Representation** — HOG
**Model** - SVM
**Accuracy** — 97.05%
**F1-Score** - 97.04%

### Worst-performing representation

**Representation** — PCA
**Model** - Logistic Regression
**Accuracy** — 86.70%
**F1-Score** - 86.56%

* Technically, CNN has the lowest accuracy at 83.45%, but for this section specifically I'm comparing the classical feature representations, PCA is the appropriate worst-performing representation here. If I include CNN in the comparison, then CNN is the overall lowest performing configuration.

### Key observation

HOG + SVM achieved the highest classification performance, with 97.05% accuracy and a 97.04% F1-score. HOG substantially outperformed raw pixels, PCA, and GLCM, indicating that structural and shape related information was particularly useful for distinguishing tumor classes in this dataset. Interestingly, combining HOG with GLCM did not improve performance, suggesting that additional texture information did not provide enough complementary information to compensate for the increased feature complexity.

---

# Classical ML vs CNN

| Approach     | Representation   | Model |   Accuracy |   F1-Score |
| ------------ | ---------------- | ----- | ---------: | ---------: |
| Classical ML | HOG              | SVM   | **0.9705** | **0.9704** |
| CNN          | Learned features | CNN   |     0.8345 |     0.8278 |


### Observation

The best performing classical ML configuration substantially outperformed the CNN configuration tested in this experiment, achieving 97.05% accuracy compared with 83.45% for the CNN. This suggests that the handcrafted HOG representation combined with an SVM was highly effective for this particular dataset and experimental setup.

However, this result should not be interpreted as evidence that **classical machine learning is inherently superior to CNNs(do not generalize pls)**. The CNN result depends on factors such as network architecture, training strategy, preprocessing, hyperparameters, regularization, data augmentation, and number of training epochs. The comparison therefore demonstrates that the tested HOG + SVM pipeline performed better than the tested CNN configuration, rather than establishing that **handcrafted features are universally better than learned representations.**

---

# Final Inference

The experiments investigated whether shape, texture, dimensionality reduction, or combinations of these representations provide the most useful information for brain MRI tumor classification using machine learning.

Among the tested approaches, **HOG + SVM produced the best performance, achieving 97.05% accuracy and 97.04% F1-score.** HOG substantially outperformed the raw-pixel baseline, suggesting that explicitly representing local gradients, edges, contours, and structural patterns provided more discriminative information than using raw grayscale intensities.

GLCM also improved upon the raw-pixel baseline, achieving 89.87% accuracy, demonstrating that texture contains useful information. However, its performance was considerably lower than HOG. Furthermore, combining HOG and GLCM reduced accuracy from 97.05% to 96.45%, indicating that adding texture features did not provide a meaningful performance benefit in this experiment.

PCA produced only a marginal improvement over raw pixels, increasing accuracy from 86.55% to 86.70%. This suggests that dimensionality reduction alone was not sufficient to substantially improve the discriminative representation.

The tested CNN achieved 83.45% accuracy, lower than all of the classical ML configurations. However, this should be interpreted cautiously: the experiment demonstrates the performance of the specific CNN architecture and training configuration used, rather than proving that CNNs are inferior to handcrafted features. But generally I would like to conclude that using HOG with SVM has the best tradeoff between classification performance and computational cost among the evaluated approaches.
