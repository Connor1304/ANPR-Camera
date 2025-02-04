import os
import cv2
import imutils
import numpy as np
import easyocr
import time
from flask import Flask, render_template, Response

# Create the Flask app
app = Flask(__name__)

# Initialize EasyOCR reader for English
reader = easyocr.Reader(['en'])

# Global variables for tracking
tracker = None
detection_interval = 1  # Run full detection every 30 frames
frame_count = 0
prev_time = time.time()

def preprocess_image(image):
    """Convert the image to grayscale, apply bilateral filtering, and perform Canny edge detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    return edged

def find_plate_contours(edged):
    """Find and return the contour that is most likely a license plate (a quadrilateral)."""
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        if len(approx) == 4:
            plate_contour = approx
            break
    return plate_contour

def extract_text_from_plate(roi):
    """Extract text from the region of interest using EasyOCR."""
    results = reader.readtext(roi)
    text = ''
    for (bbox, detected_text, prob) in results:
        if prob > 0.1:  # 80% confidence threshold
            text += detected_text + ' '
    return text.strip()

def process_frame(frame):
    """
    Process the frame using a combination of detection and tracking.
    Every 'detection_interval' frames (or if no tracker exists), run full detection.
    Otherwise, update the tracker to follow the plate.
    """
    global tracker, frame_count, prev_time
    frame_count += 1
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if tracker is None or frame_count % detection_interval == 0:
        # Run full detection
        edged = preprocess_image(frame)
        plate_contour = find_plate_contours(edged)
        if plate_contour is not None:
            x, y, w, h = cv2.boundingRect(plate_contour)
            plate_image = frame[y:y+h, x:x+w]
            plate_text = extract_text_from_plate(plate_image)
            if plate_text:
                # Initialize the tracker using a lightweight algorithm (using KCF here)
                try:
                    tracker = cv2.TrackerKCF_create()
                except AttributeError:
                    # Fallback if TrackerKCF_create is not available; try MIL
                    tracker = cv2.TrackerMIL_create()
                bbox = (x, y, w, h)
                tracker.init(frame, bbox)
                cv2.putText(frame, plate_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Save the detected plate image with the text as filename
                cv2.imwrite(f'{plate_text}.jpg', plate_image)
    else:
        # Update tracker
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            plate_image = frame[y:y+h, x:x+w]
            plate_text = extract_text_from_plate(plate_image)
            if plate_text:
                cv2.putText(frame, plate_text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            # Tracker lost the object; reset it
            tracker = None

    return frame

def gen_frames():
    """
    Generator function that captures frames from the camera, processes them,
    and yields JPEG-encoded frames for streaming.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not start camera.")
        
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Process frame (detection and tracking)
        frame = process_frame(frame)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        
        # Yield frame in a format suitable for MJPEG streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route that streams the video feed."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=2004)
