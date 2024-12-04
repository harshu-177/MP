import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from textblob import TextBlob

# Configure pytesseract path if necessary
# pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

def rotate_image(image):
    # Detect the skew angle using Hough Transform
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # Rotate the image to correct the skew
    (h, w) = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

def preprocess_image(image_path):
    # Load image and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get binary image
    _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Rotate image to correct any skew
    rotated_img = rotate_image(binary_img)
    
    # Remove noise using Gaussian Blur
    blurred = cv2.GaussianBlur(rotated_img, (5, 5), 0)
    
    # No thinning in this version
    return blurred

def segment_and_recognize(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)
    
    # Use pytesseract to extract text from the preprocessed image
    text = pytesseract.image_to_string(preprocessed_image, lang='eng')
    
    # Improve recognition accuracy using TextBlob for basic NLP correction
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    
    return corrected_text

# Example usage
image_path = 'c:\\Users\\harsh\\Downloads\\idk\\idk.webp'  # Replace with the actual image path
extracted_text = segment_and_recognize(image_path)
print("Extracted Text:", extracted_text)
