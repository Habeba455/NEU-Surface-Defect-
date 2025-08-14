import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder_path, img_size=(128, 128)):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder_path))
    
    for label_index, class_name in enumerate(class_names):
        class_folder = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_folder):
            continue
        
        for img_name in os.listdir(class_folder):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            
            img_path = os.path.join(class_folder, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(label_index)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, class_names


folder_path = '/Users/habebaftooh/industrial-defect-ml/data/NEU-DET/train/images' 
images, labels, class_names = load_images_from_folder(folder_path)


X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Train set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Classes: {class_names}")
