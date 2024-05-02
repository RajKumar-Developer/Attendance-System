import os
import cv2
import sqlite3

def detect():
    # Create a folder named 'Dataset' if it doesn't exist
    dataset_folder = 'Dataset'
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Input student roll number and name
    face_id = input('\nEnter Roll No.: ')
    student_name = input('Enter Name: ')

    # Create a folder for the student's images in 'Dataset'
    student_folder = os.path.join(dataset_folder, student_name)
    if not os.path.exists(student_folder):
        os.makedirs(student_folder)

    # Initialize SQLite database connection and cursor
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    # Create a 'students' table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS students
                      (roll_number TEXT PRIMARY KEY, name TEXT)''')
    conn.commit()  # Commit the table creation

    # # Check if the table structure needs updating
    # try:
    #     cursor.execute("ALTER TABLE students ADD COLUMN roll_number TEXT")
    # except sqlite3.OperationalError:
    #     # Column already exists, no need to alter
    #     pass

    # Insert student data into the database
    cursor.execute("INSERT INTO students (roll_number, name) VALUES (?, ?)", (face_id, student_name))
    conn.commit()  # Commit the insertion

    # Start real-time video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # Set video width
    cam.set(4, 480)  # Set video height

    # Load the Haar cascade for face detection
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print(f"\nInitializing face capture for {student_name}. Look at the camera and wait...")

    # Counter for captured face samples
    count = 0

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 204, 102), 2)
            count += 1
            face_crop = gray[y:y + h, x:x + w]

            # Save the face image to the student's folder
            img_path = os.path.join(student_folder, f"{student_name}_{count}.jpg")
            cv2.imwrite(img_path, face_crop)

        cv2.imshow('Face Capture', img)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

        if count >= 30:  # Capture 30 face samples
            break

    print("\nFace capture completed.")

    # Clean up
    cam.release()
    cv2.destroyAllWindows()

    # Close the database connection
    conn.close()

if __name__ == '__main__':
    detect()
