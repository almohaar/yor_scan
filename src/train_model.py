import os
import tensorflow as tf
from tensorflow.keras import layers, models
from config import IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, EPOCHS, ROTATION_FACTOR, ZOOM_FACTOR

# Directories for training and testing datasets.
TRAIN_DIR = 'datasets/train'
TEST_DIR = 'datasets/test'

# Create training dataset from directory
raw_train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True,
    seed=123
)

raw_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    labels='inferred',
    label_mode='int',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=False
)

# Save the class names from the dataset.
class_names = raw_train_ds.class_names
num_classes = len(class_names)
total_files = sum([len(files) for files in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, files))])
print(f"Found {total_files} files belonging to {num_classes} classes.")
print("Class names:", class_names)

# Data augmentation layer
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(ROTATION_FACTOR),
    layers.RandomZoom(ZOOM_FACTOR)
])

# Normalize pixel values to [0, 1]
normalization_layer = layers.Rescaling(1.0 / 255)

def preprocess(x, y):
    x = data_augmentation(x)
    x = normalization_layer(x)
    return x, y

train_ds = raw_train_ds.map(preprocess)
test_ds = raw_test_ds.map(lambda x, y: (normalization_layer(x), y))

# Build an improved CNN model for character classification.
model = models.Sequential([
    layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    data_augmentation,
    normalization_layer,
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS
)

# Evaluate on the test dataset
loss, acc = model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# Save the trained Keras model
os.makedirs('models', exist_ok=True)
model.save('models/yoruba_char_model.h5')
print("Saved Keras model to models/yoruba_char_model.h5")

# Convert the model to TFLite for mobile deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('models/yoruba_char_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Saved TFLite model to models/yoruba_char_model.tflite")
