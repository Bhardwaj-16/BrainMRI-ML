import os
import cv2
import numpy as np
from tqdm import tqdm

LABEL_MAP = {
    'glioma': 0,
    'meningioma': 1,
    'notumor': 2,
    'pituitary': 3
}

def load_images_from_folder(folder_path, label, target_size=(128, 128)):
    # Loads all images from a specific folder, resizes them, and converts to grayscale.

    images = []
    labels = []
    
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(folder_path, filename)
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                img_resized = cv2.resize(img, target_size)
                images.append(img_resized)
                labels.append(label)
                
    return images, labels

def load_dataset(base_path, target_size=(128, 128)):
    # Loads the entire dataset from the base path (e.g., 'data/Training')

    X = []
    y = []
    
    print(f"Loading data from {base_path}...")
    for class_name, label_idx in tqdm(LABEL_MAP.items(), desc="Classes"):
        class_folder = os.path.join(base_path, class_name)
        
        if not os.path.exists(class_folder):
            print(f"Warning: Directory {class_folder} does not exist. Skipping.")
            continue
            
        images, labels = load_images_from_folder(class_folder, label_idx, target_size)
        X.extend(images)
        y.extend(labels)
        
    return np.array(X), np.array(y)

def get_data_splits(train_path='data/Training', test_path='data/Testing', target_size=(128, 128)):
    # Loads both training and testing datasets.

    X_train, y_train = load_dataset(train_path, target_size)
    X_test, y_test = load_dataset(test_path, target_size)
    
    print(f"\nData Loading Complete:")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'data', 'Training')
    test_dir = os.path.join(base_dir, 'data', 'Testing')
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        X_train, y_train, X_test, y_test = get_data_splits(train_dir, test_dir, target_size=(128, 128))
    else:
        print("Data directories not found. Please ensure 'data/Training' and 'data/Testing' exist.")
