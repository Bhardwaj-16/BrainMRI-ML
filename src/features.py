import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops
from sklearn.decomposition import PCA
from tqdm import tqdm

def extract_hog_features(images, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    # Extracts Histogram of Oriented Gradients (HOG) features from a list of images.

    hog_features = []
    print("Extracting HOG features...")
    for image in tqdm(images):
        fd = hog(image, orientations=orientations, 
                 pixels_per_cell=pixels_per_cell, 
                 cells_per_block=cells_per_block, 
                 block_norm='L2-Hys', 
                 visualize=False)
        hog_features.append(fd)
    return np.array(hog_features)

def extract_glcm_features(images, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    # Extracts Gray Level Co-occurrence Matrix (GLCM) features.
    # Computes contrast, dissimilarity, homogeneity, energy, correlation, ASM.

    glcm_features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    
    print("Extracting GLCM features...")
    for image in tqdm(images):
        # GLCM requires integer images. Convert if necessary.
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
                
        # Compute GLCM
        glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        
        # Extract properties and flatten
        img_features = []
        for prop in properties:
            prop_vals = graycoprops(glcm, prop).flatten()
            img_features.extend(prop_vals)
            
        glcm_features.append(img_features)
        
    return np.array(glcm_features)

def combine_features(feat1, feat2):
    # Concatenates two feature sets along the feature dimension.
    return np.hstack((feat1, feat2))

def apply_pca(X_train_images, X_test_images, n_components=0.95):
    # Flattens images and applies PCA to reduce dimensionality.
    # Fits PCA on the training set and transforms both train and test sets.

    print("Flattening images for PCA...")
    X_train_flat = X_train_images.reshape(X_train_images.shape[0], -1)
    X_test_flat = X_test_images.reshape(X_test_images.shape[0], -1)
    
    print(f"Applying PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components, random_state=42)
    
    # Fit and transform on train
    X_train_pca = pca.fit_transform(X_train_flat)
    
    # Transform test
    X_test_pca = pca.transform(X_test_flat)
    
    print(f"PCA reduced feature dimension from {X_train_flat.shape[1]} to {X_train_pca.shape[1]}")
    return X_train_pca, X_test_pca, pca
