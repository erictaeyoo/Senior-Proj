import cv2
import torch
from torchvision import transforms
from PIL import Image
from DeepfakeDetector import DeepfakeDetector
import numpy as np

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepfakeDetector({})
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Open video stream
video_source = 0  # Webcam
cap = cv2.VideoCapture(video_source)

frame_rate = 5  # Process every nth frame
frame_count = 0
buffer_size = 5  # Number of frames a face can remain without being detected

# Load face detector (Haar cascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Kalman filter parameters
class KalmanFilter:
    def __init__(self, x, y, w, h):
        self.kalman = cv2.KalmanFilter(8, 4)
        self.kalman.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        self.kalman.transitionMatrix = np.eye(8, dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kalman.errorCovPost = np.eye(8, dtype=np.float32)

        # Initialize state (x, y, w, h, dx, dy, dw, dh)
        self.kalman.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32).reshape(-1, 1)

    def predict(self):
        prediction = self.kalman.predict()
        x, y, w, h = prediction[:4].flatten()
        return int(x), int(y), int(w), int(h)

    def correct(self, x, y, w, h):
        measurement = np.array([x, y, w, h], dtype=np.float32).reshape(-1, 1)
        self.kalman.correct(measurement)

# Dictionary to track faces and their Kalman filters
tracked_faces = {}  # {face_id: {'kf': KalmanFilter, 'label': 'Real', 'confidence': 0.85, 'ttl': buffer_size}}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process every nth frame for predictions
    if frame_count % frame_rate == 0:
        # Update tracked faces with new detections
        updated_faces = set()
        for (x, y, w, h) in detected_faces:
            face_id = (x, y, w, h)  # Create a temporary face ID

            # Find an existing Kalman filter close to the detected face
            matched_face_id = None
            for existing_face_id, data in tracked_faces.items():
                ex, ey, ew, eh = existing_face_id
                if abs(ex - x) < 50 and abs(ey - y) < 50:  # Threshold for matching faces
                    matched_face_id = existing_face_id
                    break

            if matched_face_id is not None:
                # Update the existing Kalman filter
                tracked_faces[matched_face_id]['kf'].correct(x, y, w, h)
                updated_faces.add(matched_face_id)
            else:
                # Create a new Kalman filter for the new face
                kf = KalmanFilter(x, y, w, h)
                tracked_faces[face_id] = {'kf': kf, 'label': None, 'confidence': 0, 'ttl': buffer_size}
                updated_faces.add(face_id)

        # Perform inference for updated faces
        for face_id in updated_faces:
            x, y, w, h = tracked_faces[face_id]['kf'].predict()

            # Preprocess face region
            face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                pred_dict = model({'image': input_tensor})
                logits = pred_dict['logits']
                probs = torch.softmax(logits, dim=1)
                is_deepfake = torch.argmax(probs, dim=1).item()
                confidence = probs[0, is_deepfake].item()

            # Update face prediction
            label = "Deepfake" if is_deepfake else "Real"
            tracked_faces[face_id]['label'] = label
            tracked_faces[face_id]['confidence'] = confidence
            tracked_faces[face_id]['ttl'] = buffer_size

    # Reduce TTL for all tracked faces
    to_delete = []
    for face_id, data in tracked_faces.items():
        if face_id not in detected_faces:
            data['ttl'] -= 1
            if data['ttl'] <= 0:  # Remove stale face
                to_delete.append(face_id)

    # Delete expired face trackers
    for face_id in to_delete:
        del tracked_faces[face_id]

    # Draw rectangles and labels for tracked faces
    for face_id, data in tracked_faces.items():
        x, y, w, h = data['kf'].predict()
        label = data['label']
        confidence = data['confidence']
        color = (0, 0, 255) if label == "Deepfake" else (0, 255, 0)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        if label is not None:
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow('Live Deepfake Detection', frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Cleanup
cap.release()
cv2.destroyAllWindows()
