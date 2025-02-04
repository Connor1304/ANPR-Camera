import cv2
import easyocr
import imutils
import time

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imshow("Gray",gray)
    edged = cv2.Canny(gray, 30, 200)
    cv2.imshow("Edged",edged)
    return edged

def find_plate_contours(edged):
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
    results = reader.readtext(roi)
    text = ''
    for (bbox, detected_text, prob) in results:
        if prob > 0.4:  # You can adjust the confidence threshold
            text += detected_text + ' '
    return text.strip()

def main():
    cap = cv2.VideoCapture(1)
    tracker = None
    detection_interval = 60  # Run full detection every 30 frames
    frame_count = 0

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optionally downscale the frame for faster processing
        frame_count += 1

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if tracker is None or frame_count % detection_interval == 0:
            edged = preprocess_image(frame)
            plate_contour = find_plate_contours(edged)
            if plate_contour is not None:
                x, y, w, h = cv2.boundingRect(plate_contour)
                plate_image = frame[y:y+h, x:x+w]
                text = extract_text_from_plate(plate_image)
                if text:
                    tracker = cv2.TrackerKCF_create()  # Using a lightweight tracker
                    bbox = (x, y, w, h)
                    tracker.init(frame, bbox)
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.imwrite(f'{text}.jpg', plate_image)
        else:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = map(int, bbox)
                plate_image = frame[y:y+h, x:x+w]
                text = extract_text_from_plate(plate_image)
                if text:
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                tracker = None

        cv2.imshow('ANPR Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()