from picamera2 import Picamera2
import time

# Initialize camera
picam2 = Picamera2()
config = picam2.create_still_configuration(main={"size": (1920, 1080)})  # 1080p for simplicity
picam2.configure(config)

try:
    # Start camera and show preview
    picam2.start(show_preview=True)
    print("Camera started, preview window open")

    # Wait 5 seconds to view preview
    time.sleep(5)

    # Capture image
    print("Capturing image")
    picam2.capture_file("test_image.jpg")
    print("Image saved as test_image.jpg")

except KeyboardInterrupt:
    print("Stopping...")
finally:
    # Cleanup
    picam2.stop()
    print("Camera stopped")
