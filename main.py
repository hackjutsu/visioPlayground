import cv2
import os

# Path to the haarcascades directory
haarcascades_dir = cv2.data.haarcascades

# Path to the haarcascade_frontalface_default.xml file
face_cascade_path = os.path.join(haarcascades_dir, 'haarcascade_frontalface_default.xml')

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Load known faces
known_faces = []
known_names = []

for filename in os.listdir('known_faces'):
    if filename.endswith('.jpg'):
        img = cv2.imread(os.path.join('known_faces', filename))
        name = os.path.splitext(filename)[0]
        known_faces.append(img)
        known_names.append(name)

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around detected faces and label known faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Compare detected face with known faces
        for i, known_face in enumerate(known_faces):
            result = cv2.matchTemplate(gray[y:y+h, x:x+w], cv2.cvtColor(known_face, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
            _, confidence, _, _ = cv2.minMaxLoc(result)

            if confidence > 0.7:
                cv2.putText(frame, known_names[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Video', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
