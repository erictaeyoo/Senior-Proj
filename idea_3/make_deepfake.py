import cv2
import dlib
import numpy as np

# Load facial landmark predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_path)

# Load target face image
target_face = cv2.imread("source_face.jpg")
target_face_gray = cv2.cvtColor(target_face, cv2.COLOR_BGR2GRAY)
target_landmarks = face_detector(target_face_gray)
if len(target_landmarks) == 0:
    raise ValueError("No face detected in target_face.jpg")
target_landmarks = shape_predictor(target_face_gray, target_landmarks[0])

# Extract landmarks from the target face
target_points = np.array([(p.x, p.y) for p in target_landmarks.parts()], dtype=np.int32)
h, w, _ = target_face.shape
target_mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillConvexPoly(target_mask, target_points, 255)

target_face_rect = cv2.boundingRect(target_points)
target_face_cropped = target_face[target_face_rect[1]:target_face_rect[1]+target_face_rect[3],
                                  target_face_rect[0]:target_face_rect[0]+target_face_rect[2]]
target_points_cropped = target_points - [target_face_rect[0], target_face_rect[1]]

# Start webcam
cap = cv2.VideoCapture(0)

def extract_landmarks(frame, gray_frame):
    faces = face_detector(gray_frame)
    if len(faces) == 0:
        return None, None
    landmarks = shape_predictor(gray_frame, faces[0])
    points = np.array([(p.x, p.y) for p in landmarks.parts()], dtype=np.int32)
    return faces[0], points

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face, landmarks = extract_landmarks(frame, gray_frame)

    if landmarks is not None:
        face_mask = np.zeros_like(gray_frame)
        cv2.fillConvexPoly(face_mask, landmarks, 255)

        face_rect = cv2.boundingRect(landmarks)
        face_cropped = frame[face_rect[1]:face_rect[1]+face_rect[3], face_rect[0]:face_rect[0]+face_rect[2]]
        landmarks_cropped = landmarks - [face_rect[0], face_rect[1]]

        # Warp target face to match the detected face
        warp_matrix = cv2.getAffineTransform(np.float32(target_points_cropped[:3]),
                                             np.float32(landmarks_cropped[:3]))
        warped_target_face = cv2.warpAffine(target_face_cropped, warp_matrix, (face_rect[2], face_rect[3]))

        # Blend the warped face with the original frame
        frame[face_rect[1]:face_rect[1]+face_rect[3], face_rect[0]:face_rect[0]+face_rect[2]] = warped_target_face

    # Show live stream with face swapping
    cv2.imshow("Live Face Swap", frame)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
