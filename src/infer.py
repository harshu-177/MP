import tensorflow as tf
import cv2
import numpy as np
from model import build_model

MODEL_PATH = "outputs/model.h5"
IMG_HEIGHT, IMG_WIDTH = 128, 512

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_PATH)
    image_path = "../img2.jpg"
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    decoded = tf.keras.backend.ctc_decode(prediction, input_length=[prediction.shape[1]])[0][0].numpy()
    predicted_text = ''.join([chr(c) for c in decoded if c != -1])
    print(f"Predicted Text: {predicted_text}")
