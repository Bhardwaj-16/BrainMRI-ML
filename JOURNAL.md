# Phase 1 - Data Prep
This phase/step is for converting our MRI data into data that our ML Code could work with.
the data_loader.py process the images and converts them into 128*128 grayscale images and saves them temporarily into the volatile memory(RAM).
Thats basically it!

# Phase 2 - Feature Extraction
Our previous stage loaded processed images into the RAM, now we take the images from RAM and convert them to numerical features that algorithms can learn from.
Im gonna implement 3 feature approaches: HOG, GLCM, PCA
HOG(Histogram of Oriented Gradient) detects shapes and edges
GLCM(Gray Level Co occurence Matrix) detects texture
HOG + GLCM
PCA(Principal Component Analysis) does diensionality reduction.

Now thsi might all seems as just fluff/fancy ML terms but they are actually really important and useful if we go deeper into detecting anomalies in Brain MRI scans.
For detecting anomailes our question is - what information in ththe image pixels actually helps distinguish glioma, meningioma, pituitary tumor, or no tumor?

Okay so now lets briefly understand what each feature extraction method does?
HOG - as I said before, it detects shapes and edges. Technically HOG represents local gradient magnitude and orientation distributions.
What does this mean for us? Take a look at this
```
Normal tissue
████████████████
██████      ████
████        ████
███  TUMOR   ███
████        ████
██████      ████
████████████████
```

The boundary might create a particular pattern of intensity changes.
The raw pixels contain all of this information, but a classifier has to figure out which pixel relationships matter.
HOG explicitly extracts information about local gradients and edge orientations.
So, instead of 16,384 raw pixel values, we get something like(this isnt what we get, its more of like computer language but for an understanding we get something which says this):
"there are strong horizontal edges here"
"there are diagonal boundaries here"
So inshort, HOG captures shape and edge information.

GLCM detects texture. It looks at relationships between neighboring pixel intensities. It summarizes spatial relationships between gray level intensities.
Example:
```
Region A:

100 101 100 101
101 100 101 100
100 101 100 101
```

and 
```
Region B:

30  180  72  210
150  40  190  80
90  220  50  170
```

Region B seems more irregular than Region A.
GLCM extracts things like:
* Contrast
* Dissimilarity
* Homogeneity
* Energy
* Correlation
* ASM

Why GLCM help is because MRI tissues can have different texture/intensity patterns, and tumors can alter those patterns.

HOG + GLCM is somethign interesting. The idea is that shape alone may not be enough, and texture alone may not be enough. So we combine them both and can get a better discrimination.
And we're gonna try to test the results experimentally by testing HOG only, GLCM only, and HOG + GLCM and draw inferences from the results.

PCA is for compression. Its completely different from GLCM and HOG. PCA addresses a dimensionality problem.
A 128×128 image has 128 * 128 = 16,384 pixel values.
And for 5000 images, this becomes a HUGE feature space. PCA tries to find the most important directions of variation in that data.
Something like:

```
16,384 dimensions
       ↓
      PCA
       ↓
maybe 500–2000 dimensions
```
and all this by retaining 95% of the variance.(The actual number depends on n_components setting)
Also IMP: PCA must be fitted only on the training set or else it would causes data leakage.
Inshort, dimensionality reduction. Many people hear this term when they're studying basic Data extraction for AI. I encountered this term in 9th Grade, when the AI Curriculum was first introduced.

Lets go through the uses of dimensionality reduction quickly:
* Faster training
* Less redundancy
* Can reduce overfitting(we have many features, Our model has a lot of opportunities to learn patterns that don't generalize. Reducing dimensionality can sometimes improve generalization.)

AND THE MOST IMPORTANT THING, WHAT SI THIS PROJECT ALL ABOUT?
The Ques: Which representation of brain MRI images provides the most useful information for classifying tumor types?

| Representation | Accuracy(example) |  
| -------------- | -------:          |	
| Raw pixels     |      82%          |   
| PCA            |      89%          |   Logistic Regression		    How far can raw visual information go?    
| HOG            |      84%          |   SVM							How useful is shape information?
| GLCM           |      78%          |   Random Forest   			    How useful is texture?
| HOG + GLCM     |      86%          |   SVM							Does combining representations help?

Then we're gonna also compare Classical Machine Learning vs CNNs.
