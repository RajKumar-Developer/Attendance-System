import cv2
import numpy as np
import os

def load_face_encodings(trained_path):
    # Print the path being used to load face encodings
    print("Loading face encodings from:", trained_path)

    # Construct the full path to the face encodings file
    encodings_file = os.path.join(trained_path, 'face_encodings.npy')

    # Check if the file exists
    if not os.path.exists(encodings_file):
        print("Error: face encodings file not found!")
        return None, None
    
    # Load the face encodings and corresponding IDs
    known_face_encodings = np.load(encodings_file)
    known_face_ids = np.load(os.path.join(trained_path, 'face_ids.npy'))
    
    return known_face_encodings, known_face_ids


def recognize(trained_path):
    # Load the trained face encodings and IDs
    known_face_encodings, known_face_ids = load_face_encodings(trained_path)

    # Haarcascade Classifier used for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # Set video width
    cam.set(4, 480)  # Set video height

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Extract face region and resize for consistency
            face_region = cv2.resize(gray[y:y+h, x:x+w], (128, 128))

            # Flatten face region for recognition
            face_encoding = face_region.flatten() / 255.0  # Normalize pixel values to [0, 1]

            # Find the closest matching user ID based on face encodings
            min_distance = float('inf')
            recognized_user_id = -1

            for i, encoding in enumerate(known_face_encodings):
                distance = np.linalg.norm(face_encoding - encoding)
                if distance < min_distance:
                    min_distance = distance
                    recognized_user_id = known_face_ids[i]

            # Retrieve the recognized user name based on the recognized_user_id
            if recognized_user_id != -1:
                # Display the recognized user name
                recognized_user_name = str(recognized_user_id)
                cv2.putText(img, recognized_user_name, (x+5, y+h-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 128, 255), 2)
            else:
                cv2.putText(img, "Unknown", (x+5, y+h-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 128, 255), 2)

        cv2.imshow('camera', img)

        # Press 'ESC' to exit video capture
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    trained_path = 'Trained'
    recognize(trained_path)
