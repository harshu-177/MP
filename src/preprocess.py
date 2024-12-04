import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants
DATA_DIR = "dataset/words"
LABELS_FILE = "dataset/words.txt"
BATCH_SIZE = 32
IMG_HEIGHT = 128
IMG_WIDTH = 512
MAX_LABEL_LENGTH = 20  # Adjust based on dataset

# Initialize the tokenizer for character-based encoding
tokenizer = Tokenizer(char_level=True)

# Load the labels from the words.txt file and get the image dimensions from the filename
def load_labels():
    labels = {}
    image_dims = {}  # Store image dimensions
    with open(LABELS_FILE, "r") as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':  # Skip comments or empty lines
                continue
            parts = line.strip().split()
            word_id = parts[0]  # Key (e.g., "a01-000u-00-00")
            label = parts[-1].strip()  # The text label

            # Try to extract the dimensions (height and width) from the line safely
            try:
                img_height = int(parts[5])  # Image height from filename
                img_width = int(parts[6])  # Image width from filename
            except ValueError:
                print(f"Skipping invalid entry: {word_id} due to invalid dimensions")
                continue  # Skip the line if height or width is invalid (non-numeric)

            labels[word_id] = label
            image_dims[word_id] = (img_height, img_width)  # Store the dimensions
    tokenizer.fit_on_texts(labels.values())  # Fit the tokenizer on the labels
    return labels, image_dims


# Preprocess the image to match the expected input shape (grayscale to RGB)
def preprocess_image(image_path, img_height, img_width):
    img = Image.open(image_path).convert("L")  # Convert to grayscale ('L' mode)

    # Debugging: print the height and width before resizing
    print(f"Resizing image: {image_path[-18:]}..")

    # Check for valid dimensions
    if img_width <= 0 or img_height <= 0:
        print(f"Invalid dimensions for image: {image_path}, skipping.")
        return None

    img = img.resize((IMG_WIDTH, IMG_HEIGHT))  # Resize the image dynamically
    img = np.array(img)  # Convert to numpy array
    img = img / 255.0  # Normalize the image
    img = img.astype(np.float32)  # Ensure the type is float32
    img = np.expand_dims(img, axis=-1)  # Add channel dimension to make it (height, width, 1)
    return img


# Convert labels to integer sequences
def encode_labels(labels):
    # Convert the labels to integer sequences using the tokenizer
    label_sequences = tokenizer.texts_to_sequences(labels)
    # Pad the labels to have consistent lengths for each sequence
    return pad_sequences(label_sequences, padding='post', maxlen=MAX_LABEL_LENGTH)


# Create a dataset using TensorFlow
def create_dataset(batch_size=BATCH_SIZE):
    labels, image_dims = load_labels()
    image_paths, texts = [], []

    for word_id, label in labels.items():
        # Construct the path for each image based on word_id
        parts = word_id.split('-')
        folder_path = os.path.join(DATA_DIR, parts[0], '-'.join(parts[:2]))  # "a01/a01-000u"
        img_filename = f"{word_id}.png"  # Image filename
        img_path = os.path.join(folder_path, img_filename)  # Full image path

        if os.path.isfile(img_path):  # Check if the image exists
            img_height, img_width = image_dims.get(word_id, (128, 512))  # Default size if not found
            img = preprocess_image(img_path, img_height, img_width)
            if img is not None:  # Only append valid images
                image_paths.append(img)
                texts.append(label)  # Corresponding label

    # Encode the labels to integer sequences
    encoded_labels = encode_labels(texts)

    # Check that image paths and labels are of the same length
    if len(image_paths) != len(encoded_labels):
        raise ValueError("Mismatch between number of images and labels")

    # Convert lists to numpy arrays to ensure homogeneous shapes
    image_paths = np.array(image_paths)  # Ensure the shape of images is homogeneous
    encoded_labels = np.array(encoded_labels)  # Ensure labels have consistent length

    # Create a TensorFlow dataset from the image paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, encoded_labels))

    # Function to process the path to an image
    def process_path(image, label):
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        label = tf.convert_to_tensor(label, dtype=tf.int32)
        return image, label

    # Map the processing function and batch the dataset
    dataset = dataset.map(process_path).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# Main function to create and print the dataset
if __name__ == "__main__":
    dataset = create_dataset()
    for images, labels in dataset.take(1):  # Take a batch of images and labels
        print(images.shape, labels)
