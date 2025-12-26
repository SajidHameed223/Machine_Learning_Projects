import cv2
from tensorflow.keras.models import model_from_json
import numpy as np
import os

# -----------------------------
# Load model
# -----------------------------
json_path = os.path.join("Face Emotions Detactor", "emotiondetector.json")
weights_path = os.path.join("Face Emotions Detactor", "emotiondetector.h5")

print("Loading model...")
with open(json_path, "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights(weights_path)
print("Model loaded successfully")

# -----------------------------
# Haar cascade
# -----------------------------
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
print("Cascade loaded:", not face_cascade.empty())

# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# -----------------------------
# Webcam setup
# -----------------------------
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Webcam not accessible")
    exit()

labels = {
    0: 'angry', 1: 'disgust', 2: 'fear',
    3: 'happy', 4: 'natural', 5: 'sad', 6: 'surprise'
}

print("Starting webcam feed... Press 'q' to quit.")

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, im = webcam.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    print("Faces detected:", len(faces))

    if len(faces) == 0:
        cv2.putText(im, "No face detected", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for (p, q, r, s) in faces:
        image = gray[q:q+s, p:p+r]
        cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)

        try:
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            print("Prediction raw:", pred)

            prediction_label = labels[pred.argmax()]
            cv2.putText(im, prediction_label, (p-10, q-10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        except Exception as e:
            print("Prediction error:", e)

    cv2.imshow("Output", im)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()