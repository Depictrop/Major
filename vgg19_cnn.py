import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Adam


def load_and_preprocess_data(directory_path, target_shape=(224, 224)):
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
val_split = 0.2
test_split = 0.1


images_array, labels_array = load_and_preprocess_data(data_directory)

# Split the data using train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(images_array, labels_array, test_size=val_split + test_split, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_split/(val_split + test_split), random_state=42)


label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)


base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer


model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train_encoded, epochs=10, validation_data=(X_val, y_val_encoded))


loss, accuracy = model.evaluate(X_test, label_encoder.transform(y_test))
print(f"Test Accuracy: {accuracy * 100:.2f}%")
