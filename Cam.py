import cv2

# Initialize the camera (use /dev/video0 for the IMX477)
cap = cv2.VideoCapture(0)  # 0 is usually the default camera device

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set resolution (optional, adjust based on your needs)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Camera opened successfully. Press 'q' to quit.")

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Display the frame
        cv2.imshow('Arducam IMX477 Test', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")
