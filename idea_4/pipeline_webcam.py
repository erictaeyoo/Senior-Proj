import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from DeepfakeDetector_conv import DeepfakeDetector
import os

# ------------------ Device Setup ------------------
device = torch.device("mps" if torch.backends.mps.is_available() 
                      else "cuda" if torch.cuda.is_available() 
                      else "cpu")

# ------------------ Load Model ------------------
model = DeepfakeDetector({
    "model_name": "resnet18",
    "pretrained": False,
    "num_classes": 2
})
model.to(device)

checkpoint = torch.load("best_model_conv.pth", map_location=device)
if 'model' in checkpoint:
    checkpoint = checkpoint['model']
model.load_state_dict(checkpoint, strict=False)
model.eval()

# ------------------ Image Preprocessing ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ------------------ Face Detection ------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ------------------ Video Source ------------------
cap = cv2.VideoCapture(1)  # Webcam
#video_path = 'asian.mp4'
#cap = cv2.VideoCapture(video_path)

# ------------------ Inference Loop ------------------
frame_count = 0
predict_every_n_frames = 1  # Update prediction every N frames

# Keep track of the last prediction
cached_faces = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % predict_every_n_frames == 0:
        cached_faces = []
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in detected_faces:
            pad = int(0.1 * w)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_dict = model({'image': input_tensor})
                logits = pred_dict['logits']
                probs = torch.softmax(logits, dim=1)
                is_deepfake = torch.argmax(probs, dim=1).item()
                confidence = probs[0, is_deepfake].item()

            label = "Deepfake" if is_deepfake else "Real"
            color = (0, 0, 255) if is_deepfake else (0, 255, 0)
            cached_faces.append({
                'box': (x, y, w, h),
                'label': label,
                'confidence': confidence,
                'color': color
            })

    # Draw cached predictions
    for face in cached_faces:
        x, y, w, h = face['box']
        label = face['label']
        confidence = face['confidence']
        color = face['color']

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    cv2.imshow('Deepfake Detection (Smoothed)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
