Python 3.13.1 (tags/v3.13.1:0671451, Dec  3 2024, 19:06:28) [MSC v.1942 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
tf.random.set_seed(42)

# Load dataset (assuming you have a CSV file with image paths and labels)
data = pd.read_csv('diabetic_retinopathy_dataset.csv')  # CSV with columns: 'image_path', 'label'
image_paths = data['image_path'].values
labels = data['label'].values

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Define image dimensions and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'image_path': X_train, 'label': y_train}),
    x_col='image_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=pd.DataFrame({'image_path': X_val, 'label': y_val}),
    x_col='image_path',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw'
)

# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Create the model
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)  # RGB images
num_classes = 5  # 5 severity levels
model = create_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
...     steps_per_epoch=len(train_generator),
...     epochs=10,
...     validation_data=val_generator,
...     validation_steps=len(val_generator)
... )
... 
... # Evaluate the model
... val_loss, val_accuracy = model.evaluate(val_generator)
... print(f"Validation Loss: {val_loss}")
... print(f"Validation Accuracy: {val_accuracy}")
... 
... # Plot training history
... def plot_training_history(history):
...     plt.figure(figsize=(12, 4))
...     
...     plt.subplot(1, 2, 1)
...     plt.plot(history.history['loss'], label='Training Loss')
...     plt.plot(history.history['val_loss'], label='Validation Loss')
...     plt.title('Loss')
...     plt.xlabel('Epoch')
...     plt.ylabel('Loss')
...     plt.legend()
...     
...     plt.subplot(1, 2, 2)
...     plt.plot(history.history['accuracy'], label='Training Accuracy')
...     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
...     plt.title('Accuracy')
...     plt.xlabel('Epoch')
...     plt.ylabel('Accuracy')
...     plt.legend()
...     
...     plt.show()
... 
... plot_training_history(history)
... 
... # Save the model
... model.save('diabetic_retinopathy_cnn_model.h5')
>>> [DEBUG ON]
>>> [DEBUG OFF]
