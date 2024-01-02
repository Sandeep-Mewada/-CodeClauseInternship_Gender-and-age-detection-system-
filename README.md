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

