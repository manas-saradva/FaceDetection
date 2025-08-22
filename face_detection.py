import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.losses import MeanAbsoluteError
from PIL import Image

# Load the model
model = load_model("C:/Users/manas/Downloads/Major project/Major project/best_model.h5", compile=False)
model.compile(loss=['binary_crossentropy', MeanAbsoluteError()], optimizer='adam', metrics=['accuracy'])

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to extract features
def extract_features(images):
    features = []
    for image in images:
        img = image.convert('L')  # Convert to grayscale
        img = img.resize((128, 128), Image.LANCZOS)
        img = img_to_array(img) / 255.0  # Normalize
        features.append(img)

    features = np.array(features)
    features = features.reshape(len(features), 128, 128, 1)
    return features

# Function to detect faces
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_images = [frame[y:y+h, x:x+w] for (x, y, w, h) in faces]
    return faces, face_images

# Function to categorize age into bins
def categorize_age(age):
    lower_bound = (age // 10) * 10 + 1
    upper_bound = lower_bound + 9
    return f"{lower_bound}-{upper_bound}"

# Initialize webcam
cap = cv2.VideoCapture(0)

frame_count = 0  # To slow down updates
age_final, gender_final = "Unknown", "Unknown"  # Default values

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces, face_images = detect_faces(frame)

    if frame_count % 10 == 0:  # Process every 10 frames
        for (x, y, w, h), face in zip(faces, face_images):
            pil_image = Image.fromarray(face)
            preprocessed_face = extract_features([pil_image])

            # Make predictions
            gender_pred, age_pred = model.predict(preprocessed_face)

            # **Instant gender classification**
            gender_final = 'Male' if gender_pred < 0.5 else 'Female'

            # **Categorized age bin**
            age_final = categorize_age(int(age_pred))

    # Display results
    for (x, y, w, h) in faces:
        cv2.putText(frame, f'Age: {age_final}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Gender: {gender_final}', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # Show webcam output
    cv2.imshow('Webcam', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1  # Increment frame count

cap.release()
cv2.destroyAllWindows()
