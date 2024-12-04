import os
from google.cloud import vision
from google.oauth2 import service_account

def extract_text_from_image(image_path):
    """
    Extracts text from a given image using Google Cloud Vision API.
    :param image_path: Path to the image file.
    :return: Extracted text.
    """
    # Load credentials and initialize the client
    credentials = service_account.Credentials.from_service_account_file("ocr-handwriting-project-35f6b0e8fe25.json")
    client = vision.ImageAnnotatorClient(credentials=credentials)

    # Read the image file
    with open(image_path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    # Perform text detection
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if not texts:
        print("No text found in the image.")
        return ""

    # The first annotation contains the full text
    extracted_text = texts[0].description
    print("Extracted Text:")
    print(extracted_text)

    return extracted_text

if __name__ == "__main__":
    image_path = input("Enter the path to the handwritten image: ").strip()
    if os.path.exists(image_path):
        extract_text_from_image(image_path)
    else:
        print("Image file not found. Please check the path and try again.")
