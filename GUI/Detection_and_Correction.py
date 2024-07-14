import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2

# Paths to your classification and correction models
classification_model_path = 'C:/Users/USER/Downloads/GUI/Models/pixel_classification_model.h5'
correction_model_path = 'C:/Users/USER/Downloads/GUI/Models/deeper_srcnn_model.h5'

# Load models
classification_model = load_model(classification_model_path)
srcnn_model = load_model(correction_model_path)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classification and Correction")
        self.root.attributes('-fullscreen', True)
        
        bg_image = Image.open('C:/Users/USER/Downloads/GUI/6114100.jpg')
        bg_image = bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_image)
        self.bg_label = tk.Label(root, image=bg_photo)
        self.bg_label.image = bg_photo
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image, font=("Helvetica", 18))
        self.upload_btn.place(relx=0.02, rely=0.02, relwidth=0.2, relheight=0.08)
        
        self.classify_btn = tk.Button(root, text="Classify", command=self.classify_image, font=("Helvetica", 18))
        self.classify_btn.place(relx=0.24, rely=0.02, relwidth=0.2, relheight=0.08)
        
        self.correct_btn = tk.Button(root, text="Correct", command=self.correct_image_display, font=("Helvetica", 18))
        self.correct_btn.place(relx=0.46, rely=0.02, relwidth=0.2, relheight=0.08)
        
        self.close_btn = tk.Button(root, text="Close", command=root.quit, font=("Helvetica", 18))
        self.close_btn.place(relx=0.68, rely=0.02, relwidth=0.2, relheight=0.08)
        
        self.result_label = tk.Label(root, text="", font=("Helvetica", 24), bg='white')
        self.result_label.place(relx=0.27, rely=0.12, relwidth=0.46, relheight=0.08)
        
        self.image_label = tk.Label(root, bg='white')
        self.image_label.place(relx=0.27, rely=0.22, relwidth=0.71, relheight=0.76)
        
        self.image_path = None
    
    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            img = Image.open(self.image_path)
            img.thumbnail((800, 600))
            img = ImageTk.PhotoImage(img)
            self.image_label.configure(image=img)
            self.image_label.image = img
            self.result_label.config(text="")
    
    def classify_image(self):
        if self.image_path:
            result = classify_image(self.image_path, classification_model)
            self.result_label.config(text=f"Classification: {result}")
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")
    
    def correct_image_display(self):
        if self.image_path:
            corrected_img = correct_image(self.image_path, srcnn_model)
            corrected_img = corrected_img.resize((800, 600), Image.LANCZOS)
            corrected_img = ImageTk.PhotoImage(corrected_img)
            self.image_label.configure(image=corrected_img)
            self.image_label.image = corrected_img
        else:
            messagebox.showwarning("Warning", "Please upload an image first.")

def classify_image(image_path, model):
    img = preprocess_image(image_path)
    prediction = model.predict(img)[0][0]
    threshold = 0.8
    if prediction >= threshold:
        return "Pixelated"
    else:
        return "High resolution"

def preprocess_image(image_path, img_height=224, img_width=224):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def correct_image(image_path, model, target_size=(800, 600)):
    img = preprocess_image(image_path)
    corrected_image = upscale_image(model, img)
    corrected_image = corrected_image.squeeze()
    corrected_image = np.clip(corrected_image, 0, 255).astype('uint8')
    
    corrected_image = Image.fromarray(corrected_image)
    corrected_image = corrected_image.resize(target_size, Image.LANCZOS)
    
    return corrected_image

def upscale_image(model, image):
    generated_image = model.predict(image)
    generated_image = np.clip(generated_image[0], 0.0, 1.0)
    generated_image = (generated_image * 255.0).astype(np.uint8)
    return generated_image

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
