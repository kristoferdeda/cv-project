import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def load_dataset(dataset_path):
    X, Y = [], []

    for label in sorted(os.listdir(dataset_path)):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path) or not label.isdigit():
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (100, 100))
            img = img / 255.0 
            img = img.reshape(100, 100, 1)  
            X.append(img)

            label_onehot = np.zeros(10)
            label_onehot[int(label)] = 1
            Y.append(label_onehot)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return shuffle(X, Y)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", input_shape=(100, 100, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    return model

def train():
    dataset_path = "./Dataset"
    X, Y = load_dataset(dataset_path)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

    model = build_model()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=32)

    model.save("model.h5")
    print("? Model saved as model.h5")
if __name__ == "__main__":
    train()
