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
pip install opencv-python
pip install dlib

import cv2
import dlib

# Load the pre-trained face detector
face_detector = dlib.get_frontal_face_detector()

# Load the pre-trained age and gender model
age_gender_detector = cv2.dnn.readNetFromCaffe(
    "deploy_age_gender.prototxt", "age_gender_model.caffemodel"
)

# Load an image
image = cv2.imread("path/to/your/image.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_detector(gray)

for face in faces:
    # Extract the face region
    (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
    face_roi = image[y:y + h, x:x + w]

    # Preprocess the face for age and gender prediction
    blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Forward pass for age and gender prediction
    age_gender_detector.setInput(blob)
    predictions = age_gender_detector.forward()

    # Get the predicted age and gender
    age = predictions[0][0]
    gender = "Male" if predictions[1][0] > 0.5 else "Female"

    # Display the results
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f"Age: {int(age)} Gender: {gender}"
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Display the final result
cv2.imshow("Gender and Age Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
import face_recognition
import cv2
import math

# Load the image
image_path = "path/to/your/image.jpg"
image = cv2.imread(image_path)

# Find face locations in the image
face_locations = face_recognition.face_locations(image)

# Load the pre-trained age and gender model
age_gender_model = cv2.dnn.readNetFromCaffe(
    "deploy_age_gender.prototxt", "age_gender_model.caffemodel"
)

# Define the age and gender classes
age_classes = [
    "< 18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"
]
gender_classes = ["Male", "Female"]

# Loop through each detected face
for (top, right, bottom, left) in face_locations:
    # Extract the face region
    face_roi = image[top:bottom, left:right]

    # Preprocess the face for age and gender prediction
    blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

    # Forward pass for age and gender prediction
    age_gender_model.setInput(blob)
    predictions = age_gender_model.forward()

    # Get the predicted age and gender
    age_index = int(math.floor(predictions[0].argmax() / len(age_classes)))
    gender_index = predictions[1].argmax()

    predicted_age = age_classes[age_index]
    predicted_gender = gender_classes[gender_index]

    # Display the results
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    label = f"Age: {predicted_age}, Gender: {predicted_gender}"
    cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Display the final result
cv2.imshow("Gender and Age Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
