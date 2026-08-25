# BrainMRI - ML
This project explores the implementation of ClassicalML for Differentiating Brain Tumor from MRI Scans. The types of tumor i'm training the model for are glioma, meningioma, pituitary & normal brain without any tumor. There are 7023 images in the dataset. glioma - 300, meningioma - 306, pituitary - 300, no tumor - 405.

Experiments:

HOG									SVM							How useful is shape information?
GLCM							Random Forest   				How useful is texture?
HOG + GLCM					        SVM							Does combining representations help?
PCA pixels						Logistic Regression		        How far can raw visual information go?
