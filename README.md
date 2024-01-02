# -CodeClauseInternship_Gender-and-age-detection-system-
Build a gender and age detection system: Collect diverse image data, preprocess, use pre-trained deep learning model, train specific models, evaluate, integrate into apps, enable real-time inference, post-process for accuracy, optimize, and optionally design a user-friendly interface. Prioritize ethics and regular updates for reliability.
from deepface import DeepFace

# Load a sample image
img_path = "path/to/your/image.jpg"

# Perform face analysis
result = DeepFace.analyze(img_path)

# Extract gender and age predictions
gender = result["gender"]
age = result["age"]

print(f"Gender: {gender}, Age: {age}")

import face_recognition
from PIL import Image, ImageDraw

def detect_age_and_gender(image_path):
    # Load the image with face_recognition
    image = face_recognition.load_image_file(image_path)

    # Find all face locations in the image
    face_locations = face_recognition.face_locations(image)

    # Load a pre-trained model for age and gender estimation
    face_encodings = face_recognition.face_encodings(image)
    
    # Create a Pillow Image to draw on
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    # Loop through each face found in the image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Use face_recognition to predict age and gender
        # Note: This is a simple example; you may want to use a more accurate pre-trained model for production
        age = 25  # Replace with actual age prediction
        gender = "Male"  # Replace with actual gender prediction

        # Draw a rectangle around the face
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=2)

        # Draw age and gender information
        draw.text((left, top - 20), f"Age: {age}", fill=(0, 255, 0))
        draw.text((left, top - 40), f"Gender: {gender}", fill=(0, 255, 0))

    # Display the result
    pil_image.show()

# Example usage
image_path = "path/to/your/image.jpg"
detect_age_and_gender(image_path)
