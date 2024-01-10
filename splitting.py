import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(directory_path, target_shape=(1024, 1024)):
    images = []
    labels = []

    for label in os.listdir(directory_path):
        label_path = os.path.join(directory_path, label)
        if os.path.isdir(label_path):
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                image = cv2.imread(image_path)
                # Convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Resize image to target shape
                image = cv2.resize(image, target_shape)
                images.append(image)
                labels.append(label)

    images_array = np.array(images)
    labels_array = np.array(labels)

    return images_array, labels_array


data_directory = 'C:\\Users\\dheer\\Desktop\\Folder Handler\\Major\\Required Features'
train_directory = 'C:\\Users\\dheer\\Desktop\\Folder Handler\\Major\\train'
val_directory = 'C:\\Users\\dheer\\Desktop\\Folder Handler\\Major\\val'
test_directory = 'C:\\Users\\dheer\\Desktop\\Folder Handler\\Major\\test'
val_split = 0.2
test_split = 0.1


images_array, labels_array = load_and_preprocess_data(data_directory)

X_train, X_temp, y_train, y_temp = train_test_split(images_array, labels_array, test_size=val_split + test_split, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_split/(val_split + test_split), random_state=42)


def save_to_directory(images, labels, directory):
    for label in np.unique(labels):
        label_indices = np.where(labels == label)[0]
        for idx in label_indices:
            image = images[idx]
            label_directory = os.path.join(directory, label)
            os.makedirs(label_directory, exist_ok=True)
            cv2.imwrite(os.path.join(label_directory, f"{idx}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


save_to_directory(X_train, y_train, train_directory)
save_to_directory(X_val, y_val, val_directory)
save_to_directory(X_test, y_test, test_directory)

print("Data split and saved to directories successfully!")
