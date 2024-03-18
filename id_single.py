import cv2
import os
import numpy as np

# Load your photo
your_photo = cv2.imread("reference_photos/ugo.jpg", cv2.IMREAD_GRAYSCALE)

# Create a face detector object
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create a video capture object
video_capture = cv2.VideoCapture(0)

# Set the desired frame rate (e.g., 10 frames per second)
# desired_fps = 10.0
# video_capture.set(cv2.CAP_PROP_FPS, desired_fps)

# 1- Real-time applications: For real-time applications like video conferencing, augmented reality, or live video processing, a higher frame rate (e.g., 25-30 FPS) is generally preferred to ensure smooth and natural motion.
# 2- Object tracking and motion analysis: If you need to track moving objects or analyze motion patterns accurately, a higher frame rate (e.g., 25-30 FPS or higher) is recommended to capture the necessary detail and avoid missing important events.
# 3- Face recognition and detection: For face recognition and detection tasks, a lower frame rate (e.g., 10-15 FPS) may be sufficient, as facial features don't typically change rapidly between frames.
# 4- Computational resources: Higher frame rates require more computational power for processing. If you have limited hardware resources (e.g., low-end CPU or embedded systems), a lower frame rate (e.g., 5-10 FPS) may be more appropriate to balance performance and resource utilization.

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Skip processing if the frame is not read correctly
    if not ret:
        continue

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    # Detect faces in the frame
    # The detectMultiScale() function is used to detect objects (in this case, faces) within an image or video frame.
    # gray: This is the grayscale image or frame in which you want to detect faces. Haar Cascade classifiers work best with grayscale images.
    # scaleFactor=1.1: This parameter specifies the scale factor used for scaling the image during the detection process. A value greater than 1 increases the number of scales, allowing the detector to detect smaller objects, but also increasing the computation time. A good starting value is often 1.1 or 1.2.
    # minNeighbors=5: This parameter specifies the minimum number of neighboring rectangles that need to be detected for a region to be considered a valid object (in this case, a face). Increasing this value can help reduce false positives but may also miss some objects.
    # minSize=(30, 30): This parameter specifies the minimum size of the object (in pixels) that you want to detect. Any detected object smaller than this size will be ignored. Setting this value appropriately can help filter out small, irrelevant detections and improve performance.
    # The detectMultiScale() function returns a list of rectangles (faces) containing the coordinates (x, y, width, height) of the detected objects (faces) in the input image or frame.
    # You can adjust these parameters to fine-tune the face detection process based on your specific requirements and the quality of your input images or video frames. For example, increasing the minNeighbors value can reduce false positives, while decreasing the minSize value can allow the detection of smaller faces.
    # It's worth noting that the performance and accuracy of the face detection process also depend on the quality of the pre-trained Haar Cascade classifier used (haarcascade_frontalface_default.xml in this case). OpenCV provides several pre-trained classifiers for different object types, and you can also train your own custom classifiers if needed.
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Extract the face from the frame
        face_roi = gray[y:y+h, x:x+w]

        # Compare the face with your photo
        result = cv2.matchTemplate(face_roi, your_photo, cv2.TM_CCOEFF_NORMED)
        similarity = np.max(result)

        # If the face matches your photo, draw a green rectangle and display "Match" text
        if similarity > 0.7:  # Adjust the similarity threshold as needed
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Match", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # If the face doesn't match your photo, draw a red rectangle and display "Not Matched" text
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Not Matched", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("Face Recognition", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()

'''
Function	                                                        Description	                            Parameters
cv2.imread(filename, flags)	                                        Load an image from a file	            filename (str): Path to the image file, flags (int): Way to load the image (e.g., cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE)
cv2.imshow(window_name, image)	                                    Display an image in a window	        window_name (str): Name of the window, image (numpy.ndarray): Image to be displayed
cv2.waitKey(delay)	                                                Wait for a key press event	            delay (int): Delay in milliseconds (0 for infinite delay)
cv2.destroyAllWindows()	                                            Close all windows	                    -
cv2.VideoCapture(index)                                         	Create a video capture object	        index (int or str): Camera index (0 for default camera) or video file path
cv2.CascadeClassifier(filename)                                 	Create a face detector object	        filename (str): Path to the pre-trained classifier file (e.g., "haarcascade_frontalface_default.xml")
cv2.cvtColor(image, code)	                                        Convert the image color space	        image (numpy.ndarray): Input image, code (int): Color space conversion code (e.g., cv2.COLOR_BGR2GRAY)
cv2.rectangle(image, start, end, color, thickness)	                Draw a rectangle on an image	        image (numpy.ndarray): Image to draw on, start (tuple): Starting coordinates (x, y), end (tuple): Ending coordinates (x, y), color (tuple): Rectangle color (B, G, R), thickness (int): Thickness of the rectangle line (negative for filled)
cv2.putText(image, text, org, font, fontScale, color, thickness)	Draw text on an image	                image (numpy.ndarray): Image to draw on, text (str): Text to be drawn, org (tuple): Starting coordinates (x, y), font (int): Font type (e.g., cv2.FONT_HERSHEY_SIMPLEX), fontScale (float): Font scale factor, color (tuple): Text color (B, G, R), thickness (int): Thickness of the text line
cv2.matchTemplate(image, templ, method)	                            Perform template matching	            image (numpy.ndarray): Source image, templ (numpy.ndarray): Template image, method (int): Matching method (e.g., cv2.TM_CCOEFF_NORMED)
cv2.VideoCapture(index).read()	                                    Capture a frame from the video      	- Returns: ret (bool): True if frame is read correctly, False otherwise, frame (numpy.ndarray): Captured frame
'''