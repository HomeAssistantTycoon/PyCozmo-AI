import pycozmo
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import os
import sys

# ---------------- CONFIG ----------------
MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.5
# ---------------------------------------


def main():
    # Verify model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found in repo root.")
        sys.exit(1)

    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model (OFFLINE)
    model = YOLO(MODEL_PATH)

    latest_frame = None
    running = True

    def on_camera_image(cli, image):
        nonlocal latest_frame

        frame = np.array(image)

        # Cozmo camera is grayscale
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        latest_frame = frame

    # Connect to Cozmo
    with pycozmo.connect() as cli:
        cli.enable_camera()
        cli.add_handler(pycozmo.event.EvtNewRawCameraImage, on_camera_image)

        # REQUIRED on Windows
        cv2.namedWindow("Cozmo YOLO", cv2.WINDOW_NORMAL)

        print("Cozmo YOLO live detection")
        print("CLICK THE WINDOW ONCE")
        print("Press Q to quit")

        try:
            while running:
                if latest_frame is not None:
                    # Run YOLO
                    results = model(
                        latest_frame,
                        conf=CONF_THRESHOLD,
                        device=device,
                        verbose=False
                    )

                    annotated = results[0].plot()
                    cv2.imshow("Cozmo YOLO", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False

                time.sleep(0.01)

        except KeyboardInterrupt:
            pass

        finally:
            print("Shutting down cleanly...")
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
