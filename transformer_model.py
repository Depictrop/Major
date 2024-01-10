from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


train_directory = 'C:\\Users\\dheer\\Desktop\\Folder Handler\\Major\\train'
val_directory = 'C:\\Users\\dheer\\Desktop\\Folder Handler\\Major\\val'
test_directory = 'C:\\Users\\dheer\\Desktop\\Folder Handler\\Major\\test'


batch_size = 32
patch_size = 224  

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


input_layer = Input(shape=(patch_size, patch_size, 3))
flatten_layer = Flatten()(input_layer)
output_layer = Dense(1, activation='sigmoid')(flatten_layer) 
vit_model = Model(inputs=input_layer, outputs=output_layer)


vit_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


history = vit_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)


test_loss, test_acc = vit_model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")


model_save_path = 'vision_transformer_model.h5'
vit_model.save(model_save_path)
print(f"Model saved to {model_save_path}")
