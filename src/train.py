import tensorflow as tf
from preprocess import create_dataset
from model import build_model

# Constants
BATCH_SIZE = 32
EPOCHS = 20
OUTPUT_PATH = "outputs/model.h5"

if __name__ == "__main__":
    # Load the dataset
    train_dataset = create_dataset(batch_size=BATCH_SIZE)

    # Build the model
    model = build_model(img_width=512, img_height=128, num_chars=80)

    # Train the model
    model.fit(train_dataset, epochs=EPOCHS)

    # Save the model after training
    model.save(OUTPUT_PATH)
    print(f"Model saved at {OUTPUT_PATH}")
