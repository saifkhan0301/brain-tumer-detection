import tkinter as tk
from tkinter import filedialog, messagebox
from segmentation import segment_tumor_kmeans
from cnn_model import create_cnn_model
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf


def classify_image(model, image_path):
    """Classify MRI image using trained CNN model."""
    img = cv2.imread(image_path, 0)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 128, 128, 1)

    prediction = model.predict(img)[0][0]
    label = "Tumor Detected" if prediction > 0.5 else "No Tumor"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    print(f"Prediction: {label} (Confidence: {confidence:.2%})")
    return label, confidence


class BrainTumorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Tumor Detection System")

        try:
            self.model = tf.keras.models.load_model("models/cnn_brain_tumor.h5")
        except Exception as e:
            print("Model not found or failed to load:", e)
            self.model = None
            messagebox.showwarning("Warning", "Model not found. Please train the model first.")

        # Upload Button
        self.upload_btn = tk.Button(root, text="Upload MRI Image", command=self.upload_image, font=("Arial", 12), padx=10, pady=5)
        self.upload_btn.pack(pady=10)

        # Frame for images
        self.image_frame = tk.Frame(root)
        self.image_frame.pack()

        # Original Image Panel
        self.original_label = tk.Label(self.image_frame, text="Original MRI", font=("Arial", 12, "bold"))
        self.original_label.grid(row=0, column=0, padx=10)
        self.original_panel = tk.Label(self.image_frame)
        self.original_panel.grid(row=1, column=0, padx=10)

        # Segmented Image Panel
        self.segmented_label = tk.Label(self.image_frame, text="Segmented Tumor", font=("Arial", 12, "bold"))
        self.segmented_label.grid(row=0, column=1, padx=10)
        self.segmented_panel = tk.Label(self.image_frame)
        self.segmented_panel.grid(row=1, column=1, padx=10)

        # Result Label
        self.result_label = tk.Label(root, text="", font=("Arial", 14), fg="green")
        self.result_label.pack(pady=10)

    def upload_image(self):
        if self.model is None:
            messagebox.showerror("Error", "Model not loaded. Cannot classify.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if file_path:
            segmented_path = "temp_segmented.jpg"
            try:
                segment_tumor_kmeans(file_path, segmented_path)
            except Exception as e:
                messagebox.showerror("Error", f"Segmentation failed: {e}")
                return

            try:
                label, conf = classify_image(self.model, file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Classification failed: {e}")
                return

            original = cv2.imread(file_path, 0)
            seg = cv2.imread(segmented_path, 0)

            if original is None or seg is None:
                messagebox.showerror("Error", "Failed to load images after processing.")
                return

            # Convert grayscale to RGB for display
            original_color = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
            seg_color = cv2.cvtColor(seg, cv2.COLOR_GRAY2RGB)

            # Resize for display
            original_display = cv2.resize(original_color, (256, 256))
            seg_display = cv2.resize(seg_color, (256, 256))

            # Add labels to images
            original_display = cv2.putText(original_display, 'Original', (10, 20),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            seg_display = cv2.putText(seg_display, 'Segmented', (10, 20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Display results
            original_img = Image.fromarray(original_display)
            original_tk = ImageTk.PhotoImage(original_img)
            self.original_panel.configure(image=original_tk)
            self.original_panel.image = original_tk

            segmented_img = Image.fromarray(seg_display)
            segmented_tk = ImageTk.PhotoImage(segmented_img)
            self.segmented_panel.configure(image=segmented_tk)
            self.segmented_panel.image = segmented_tk

            # Update result label
            self.result_label.config(text=f"{label} | Confidence: {conf * 100:.2f}%")


if __name__ == "__main__":
    root = tk.Tk()
    app = BrainTumorApp(root)
    root.mainloop()