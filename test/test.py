import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Dummy CNN model
model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Conv2D(8, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(4, activation='softmax')
])

model.save("maize_model.h5")
