import cv2
import os
import numpy as np

# Load your reference photos in grayscale and create a dictionary mapping photos to names
reference_photos = {}
for photo_file in os.listdir("reference_photos"):
    name = photo_file.split(".")[0]  # Assuming the name is the file name without the extension
    photo_path = os.path.join("reference_photos", photo_file)
    reference_photos[name] = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)

# Create a face detector object using the pre-trained Haar cascade classifier
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create a video capture object to access the default camera
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame from the camera
    ret, frame = video_capture.read()

    # Skip processing if the frame is not read correctly
    if not ret:
        continue

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Extract the face region of interest (ROI) from the grayscale frame
        face_roi = gray[y:y+h, x:x+w]

        # Create a label to display on the frame
        label = "Unknown"
        color = (0, 0, 255)  # Red color for unknown faces

        # Compare the face ROI with each reference photo
        for name, ref_photo in reference_photos.items():
            result = cv2.matchTemplate(face_roi, ref_photo, cv2.TM_CCOEFF_NORMED)
            similarity = np.max(result)

            # If the similarity score is above a certain threshold, consider it a match
            if similarity > 0.7:  # Adjust the similarity threshold as needed
                label = name.capitalize()
                color = (0, 255, 0)  # Green color for matched faces
                break

        # Draw a rectangle around the face and display the label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting frame
    cv2.imshow("Face Recognition", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()