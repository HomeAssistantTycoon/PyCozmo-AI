import pycozmo
import cv2
import numpy as np
import time
import os

# ---------------- CONFIG ----------------
OUTPUT_DIR = "dataset_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

running = True
latest_frame = None
image_count = 0
# ----------------------------------------


def on_camera_image(cli, image):
    global latest_frame

    frame = np.array(image)

    # Cozmo camera is grayscale
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    latest_frame = frame


with pycozmo.connect() as cli:
    cli.enable_camera()
    cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_camera_image)

    # EXPLICIT window creation (CRITICAL on Windows)
    cv2.namedWindow("Cozmo Camera", cv2.WINDOW_NORMAL)

    print("Cozmo camera live")
    print("Controls:")
    print("  S = Save image")
    print("  Q = Quit")
    print("CLICK THE CAMERA WINDOW ONCE TO ENABLE KEYS")

    try:
        while running:
            if latest_frame is not None:
                cv2.imshow("Cozmo Camera", latest_frame)

            # waitKey MUST come after imshow
            key = cv2.waitKey(1)

            if key == ord('s') and latest_frame is not None:
                filename = f"img_{image_count:05d}.jpg"
                path = os.path.join(OUTPUT_DIR, filename)

                cv2.imwrite(path, latest_frame)
                print(f"Saved {path}")
                image_count += 1

            elif key == ord('q'):
                running = False

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass

    finally:
        print("Shutting down cleanly...")
        cv2.destroyAllWindows()


