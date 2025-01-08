import cv2

def test_webcam():
    # Open a connection to the webcam (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Press 'q' to exit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture a frame.")
            break

        # Display the frame
        cv2.imshow('Webcam Test', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam()
