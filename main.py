import argparse
import os
import matplotlib.pyplot as plt
from segmentation import segment_tumor_kmeans
from cnn_model import create_cnn_model, prepare_data
import tensorflow as tf
import numpy as np
import cv2

def train_cnn():
    data_dir = "data/brain_tumor_dataset"
    (X_train, y_train), (X_test, y_test) = prepare_data(data_dir)

    model = create_cnn_model()
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    model.save("models/cnn_brain_tumor.h5")
    print("Model saved.")

    # Plot accuracy and loss
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig("results/training_logs/accuracy.png")
    plt.show()

    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.2f}, Loss: {test_loss:.2f}")

def classify_image(model, image_path):
    img = cv2.imread(image_path, 0)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 128, 128, 1)

    prediction = model.predict(img)[0][0]
    label = "Tumor Detected" if prediction > 0.5 else "No Tumor"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    print(f"Prediction: {label} (Confidence: {confidence:.2%})")
    return label, confidence

def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Detection System")
    parser.add_argument("--train", action="store_true", help="Train the CNN model")
    parser.add_argument("--image", type=str, help="Path to MRI image for segmentation/classification")
    parser.add_argument("--segment", type=str, help="Path to MRI image for segmentation only")

    args = parser.parse_args()

    if args.train:
        train_cnn()

    elif args.image:
        model = tf.keras.models.load_model("models/cnn_brain_tumor.h5")
        segmented_path = os.path.join("results/segmented_images/", os.path.basename(args.image))
        segment_tumor_kmeans(args.image, segmented_path)

        label, _ = classify_image(model, args.image)

        original = cv2.imread(args.image, 0)
        segmented = cv2.imread(segmented_path, 0)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original MRI")
        plt.imshow(original, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title(f"Segmented - {label}")
        plt.imshow(segmented, cmap='jet')
        plt.show()

    elif args.segment:
        segmented_path = os.path.join("results/segmented_images/", os.path.basename(args.segment))
        segment_tumor_kmeans(args.segment, segmented_path)

        original = cv2.imread(args.segment, 0)
        segmented = cv2.imread(segmented_path, 0)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original MRI")
        plt.imshow(original, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title("Segmented Tumor")
        plt.imshow(segmented, cmap='jet')
        plt.show()

if __name__ == "__main__":
    main()