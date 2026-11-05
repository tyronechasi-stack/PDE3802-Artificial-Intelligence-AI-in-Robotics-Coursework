"""Sketch 5: classify one main item at a time using a standard webcam."""

import argparse

import cv2

from predict_image import classify_frame, load_classifier


def live_camera(camera_number=0):
    # 1. Load our model once, not once for every frame.
    model, classes, transform = load_classifier()
    camera = cv2.VideoCapture(camera_number)
    try:
        if not camera.isOpened():
            raise OSError("Cannot open camera. Check permissions or try --camera 1.")
        print("Show one main item. Press Q in the camera window to finish.")
        while True:
            # 2. Read the next camera frame.
            ok, frame = camera.read()
            if not ok:
                raise OSError("The camera stopped supplying frames.")

            # 3. Reuse the same RGB conversion and prediction as a saved photo.
            index, confidence = classify_frame(frame, model, transform)
            name = classes[index].split("_", 1)[1].replace("_", " ")
            message = f"{name}: {confidence:.1%}"
            cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.imshow("One item at a time - Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # 4. Always free the webcam, even if an error occurs.
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    try:
        live_camera(args.camera)
    except (ValueError, OSError, cv2.error) as error:
        raise SystemExit(f"Cannot use camera: {error}")
