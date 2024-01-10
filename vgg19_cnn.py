from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras import models, layers

# Define paths
train_directory = 'C:\\Users\\dheer\\Desktop\\Folder Handler\\Major\\train'
val_directory = 'C:\\Users\\dheer\\Desktop\\Folder Handler\\Major\\val'
test_directory = 'C:\\Users\\dheer\\Desktop\\Folder Handler\\Major\\test'

# Data augmentation using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()  # No augmentation for validation/test data
test_datagen = ImageDataGenerator()

# Load VGG19 model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(1024, 1024, 3))

# Create new model on top of VGG19
num_classes = ...  # Define the number of classes based on your data
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(1024, 1024),
    batch_size=batch_size,
    class_mode='sparse'
)

val_generator = val_datagen.flow_from_directory(
    val_directory,
    target_size=(1024, 1024),
    batch_size=batch_size,
    class_mode='sparse'
)

test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=(1024, 1024),
    batch_size=batch_size,
    class_mode='sparse',
    shuffle=False
)

# Calculate steps per epoch for training
steps_per_epoch = train_generator.n // batch_size
validation_steps = val_generator.n // batch_size

# Train the model
epochs = 50
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=validation_steps
)

# Evaluate the model on test set
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the model
model.save("vgg19_model.h5")

print("Model saved successfully!")
