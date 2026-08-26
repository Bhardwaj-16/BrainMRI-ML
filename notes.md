Research Topic:
**Classical Machine Learning vs CNNs for Brain MRI Tumor Classification also which representation of brain MRI images provides the most useful information for classifying tumor types?**

We can compare:
Features              				Model								Question

HOG									SVM							How useful is shape information?
GLCM							Random Forest   				How useful is texture?
HOG + GLCM					        SVM							Does combining representations help?
PCA pixels						Logistic Regression		        How far can raw visual information go?

We gotta extract features, and then train the model based on that. Im gonna use colab to train the model. Then we can also run our experiments.
Then we can add evaluation utilities and PyTorch ResNet18 CNN baseline
