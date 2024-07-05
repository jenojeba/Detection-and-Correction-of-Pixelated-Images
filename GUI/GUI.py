import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button
from PIL import Image, ImageTk, ImageSequence
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

classification_model_path = r'C:\Users\91730\Downloads\pixel_classification_model.h5'
correction_model_path = r'C:\Users\91730\Downloads\srcnn_model.h5'

print("Loading classification model...")
classification_model = load_model(classification_model_path)
print("Classification model loaded.")

print("Loading correction model...")
correction_model = load_model(correction_model_path)
print("Correction model loaded.")


def preprocess_image(image_path, img_height=224, img_width=224):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def classify_image(image_path):
    img = preprocess_image(image_path)
    prediction = classification_model.predict(img)[0][0]
    threshold = 0.8
    print(f"Classification prediction: {prediction}")
    return prediction >= threshold

def preprocess_image_for_correction(image_path, target_size=(256, 256)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    print(f"Image array shape for correction: {img_array.shape}")
    return img_array


def correct_image(image_path):
    print("Preprocessing image for correction...")
    img_array = preprocess_image_for_correction(image_path)
    print(f"Image array shape: {img_array.shape}")
    corrected_img_array = correction_model.predict(img_array)[0]
    print(f"Corrected image array shape: {corrected_img_array.shape}")
    corrected_img_array = (corrected_img_array * 255).astype(np.uint8)
    corrected_img = Image.fromarray(corrected_img_array)
    print("Image correction complete.")
    return corrected_img

def load_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    global loaded_image_path
    loaded_image_path = file_path

    img = Image.open(file_path)
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(img)

    image_label.config(image=img)
    image_label.image = img

def classify():
    if not loaded_image_path:
        messagebox.showerror("Error", "No image loaded")
        return

    is_pixelated = classify_image(loaded_image_path)
    if is_pixelated:
        classification_label.config(text="The image is pixelated.")
    else:
        classification_label.config(text="The image is high resolution.")

def correct():
    if not loaded_image_path:
        messagebox.showerror("Error", "No image loaded")
        return

    corrected_img = correct_image(loaded_image_path)
    corrected_img = corrected_img.resize((256, 256), Image.Resampling.LANCZOS)
    corrected_img = ImageTk.PhotoImage(corrected_img)

    corrected_image_label.config(image=corrected_img)
    corrected_image_label.image = corrected_img

def update_background():
    global bg_label, bg_images, bg_image_index
    bg_image_index = (bg_image_index + 1) % len(bg_images)
    bg_label.config(image=bg_images[bg_image_index])
    root.after(100, update_background)  # Update every 100 ms

root = tk.Tk()
root.title("Image Pixelation Detector and Corrector")
root.state('zoomed')  # This maximizes the window

loaded_image_path = None


root.columnconfigure([0, 1, 2], weight=1)
root.rowconfigure([0, 1, 2], weight=1)


bg_image_path = r'C:\Users\91730\Downloads\gradient-abstract-background\6114100.jpg'  # Replace with the path to your animated background GIF
bg_image = Image.open(bg_image_path)
bg_images = [ImageTk.PhotoImage(frame.copy().resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.Resampling.LANCZOS))
             for frame in ImageSequence.Iterator(bg_image)]

bg_image_index = 0


bg_label = Label(root)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)
update_background()

button_font = ("Arial", 12, "bold")
label_font = ("Arial", 12, "bold")
button_bg = '#4CAF50'
button_fg = '#ffffff'
label_bg = '#f0f0f0'
label_fg = '#000080'


load_image_button = Button(root, text="Load Image", command=load_image, font=button_font, bg=button_bg, fg=button_fg, width=20, height=2)
load_image_button.grid(row=0, column=0, pady=10, padx=10)


image_label = Label(root, bg=label_bg)
image_label.grid(row=1, column=0, pady=10, padx=10)


classify_button = Button(root, text="Classify", command=classify, font=button_font, bg=button_bg, fg=button_fg, width=20, height=2)
classify_button.grid(row=0, column=1, pady=10, padx=10)


classification_label = Label(root, text="", font=label_font, bg=label_bg, fg=label_fg)
classification_label.grid(row=1, column=1, pady=10, padx=10)

correct_button = Button(root, text="Correct", command=correct, font=button_font, bg=button_bg, fg=button_fg, width=20, height=2)
correct_button.grid(row=0, column=2, pady=10, padx=10)

corrected_image_label = Label(root, bg=label_bg)
corrected_image_label.grid(row=1, column=2, pady=10, padx=10)

load_image_button.lift()
classify_button.lift()
correct_button.lift()
image_label.lift()
classification_label.lift()
corrected_image_label.lift()

root.mainloop()
