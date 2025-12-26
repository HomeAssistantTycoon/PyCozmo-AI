import pycozmo
import cv2
import numpy as np
import os
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = r"C:\Users\jsimo_c94fbqh\OneDrive\Desktop\Cozmo AI\models\cube_yolo3\weights\best.pt"
OUTPUT_DIR = "dataset_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

running = True
latest_frame = None
image_count = 0

# Load trained YOLO model
model = YOLO(MODEL_PATH)


# ----------------------------------------
def on_camera_image(cli, image):
    global latest_frame

    frame = np.array(image)

    # Convert grayscale to BGR
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    latest_frame = frame


def detect_and_draw(frame):
    # Run YOLO detection
    results = model(frame)  # frame can be a numpy array
    annotated_frame = results[0].plot()  # draw boxes on frame
    return annotated_frame


with pycozmo.connect() as cli:
    cli.enable_camera()
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_camera_image)

    cv2.namedWindow("Cozmo Camera", cv2.WINDOW_NORMAL)

    print("Cozmo camera live")
    print("Controls:")
    print("  S = Save image")
    print("  Q = Quit")
    print("CLICK THE CAMERA WINDOW ONCE TO ENABLE KEYS")

    try:
        while running:
            if latest_frame is not None:
                annotated = detect_and_draw(latest_frame)
                cv2.imshow("Cozmo Camera", annotated)

            key = cv2.waitKey(1)

            if key == ord('s') and latest_frame is not None:
                filename = f"img_{image_count:05d}.jpg"
                path = os.path.join(OUTPUT_DIR, filename)
                cv2.imwrite(path, latest_frame)
                print(f"Saved {path}")
                image_count += 1

            elif key == ord('q'):
                running = False

    except KeyboardInterrupt:
        pass

    finally:
        print("Shutting down cleanly...")
        cv2.destroyAllWindows()

