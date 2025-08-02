""" 
Make sure before running:

1. ls /dev/video* 

2. gst-launch-1.0 --version 

3. gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=1920,height=1080,format=NV12,framerate=30/1' ! nvvidconv ! nvoverlaysink  

"""

import cv2

# GStreamer pipeline for IMX477 on Jetson Nano
gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)

cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        cv2.imshow('Arducam IMX477 Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")

