import os
LABEL_MAP = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
for class_name in LABEL_MAP:
    class_folder = os.path.join('data/Testing', class_name)
    files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"{class_name}: {len(files)}")
