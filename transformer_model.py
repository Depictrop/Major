from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define directories
train_directory = 'C:\\Users\\dheer\\Desktop\\Folder Handler\\Major\\train'
val_directory = 'C:\\Users\\dheer\\Desktop\\Folder Handler\\Major\\val'
test_directory = 'C:\\Users\\dheer\\Desktop\\Folder Handler\\Major\\test'

# Define data generators
batch_size = 32
patch_size = 224  # Assuming square images of size 224x224

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(patch_size, patch_size),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_directory,
    target_size=(patch_size, patch_size),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=(patch_size, patch_size),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Define the Vision Transformer model (ViT)
input_layer = Input(shape=(patch_size, patch_size, 3))
# Here, you'd typically add the ViT layers, but for simplicity, we're just using a dense layer
flatten_layer = Flatten()(input_layer)
output_layer = Dense(1, activation='sigmoid')(flatten_layer)  # For binary classification
vit_model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
vit_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = vit_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# Evaluate the model on the test set
test_loss, test_acc = vit_model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the model
model_save_path = 'vision_transformer_model.h5'
vit_model.save(model_save_path)
print(f"Model saved to {model_save_path}")