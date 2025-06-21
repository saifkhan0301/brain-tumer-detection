from tensorflow.keras import layers, models
import numpy as np
import cv2
from glob import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def create_cnn_model(input_shape=(128, 128, 1)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def prepare_data(data_dir="data/brain_tumor_dataset", target_size=(128, 128)):
    images = []
    labels = []

    for label, class_name in enumerate(['no', 'yes']):
        class_dir = os.path.join(data_dir, class_name)
        for img_path in glob(os.path.join(class_dir, "*.jpg")):
            img = cv2.imread(img_path, 0)  # Grayscale
            if img is None:
                continue
            img = cv2.resize(img, target_size)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            images.append(img)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    return (X_train, y_train), (X_test, y_test)