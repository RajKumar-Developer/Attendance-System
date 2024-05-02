import os
import numpy as np
import cv2

def train():
    dataset_path = 'Dataset'

    # Lists to store face encodings and corresponding labels
    known_face_encodings = []
    known_face_ids = []

    def loadImagesAndEncode(path):
        for user_folder in os.listdir(path):
            user_folder_path = os.path.join(path, user_folder)
            if not os.path.isdir(user_folder_path):
                continue
            
            # Extract user name from the folder name
            user_name = user_folder

            # Iterate through images in the user's folder
            for file in os.listdir(user_folder_path):
                if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                    # Extract user ID from the image filename
                    parts = os.path.splitext(file)[0].split('_')
                    if len(parts) < 2:
                        print(f"Ignore image '{file}': Invalid filename format.")
                        continue
                    
                    try:
                        image_number = int(parts[-1])
                    except ValueError:
                        print(f"Ignore image '{file}': Invalid image number format.")
                        continue
                    
                    # Use the username as the ID for the user
                    user_id = user_name

                    # Load image using OpenCV
                    image_path = os.path.join(user_folder_path, file)
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Failed to load image: {image_path}")
                        continue
                    
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Use Haar cascade for face detection
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    for (x, y, w, h) in faces:
                        # Extract face region
                        face_region = gray[y:y+h, x:x+w]

                        # Resize face region to a fixed size (e.g., 128x128) for consistency
                        face_region = cv2.resize(face_region, (128, 128))

                        # Encode face by flattening and normalizing pixel values
                        face_encoding = face_region.flatten() / 255.0  # Normalize pixel values to [0, 1]

                        # Append face encoding and id to lists
                        known_face_encodings.append(face_encoding)
                        known_face_ids.append(user_id)

    print("\nTraining your data. Please wait...")
    loadImagesAndEncode(dataset_path)

    # Convert lists to numpy arrays for training
    known_face_encodings = np.array(known_face_encodings)
    known_face_ids = np.array(known_face_ids)

    # Save the trained encodings and labels to files
    trained_path = 'Trained'
    if not os.path.exists(trained_path):
        os.makedirs(trained_path)

    np.save(os.path.join(trained_path, 'face_encodings.npy'), known_face_encodings)
    np.save(os.path.join(trained_path, 'face_ids.npy'), known_face_ids)

    print("\nTraining completed. Encodings saved.")

if __name__ == '__main__':
    train()
