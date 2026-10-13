import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import math
import os

# === Configuration ===
OFFSET = 30
IMG_SIZE = 640
SAVE_DELAY = 0
MIN_CONF = 0.7

GESTURE_NAME = "Go"  # Change to "Stop" for another gesture
FOLDER = "BinBot/img_dataset/Go" #Change to "Stop" for another gesture

os.makedirs(FOLDER, exist_ok=True)


def prepare_image(img, hand):
    """
    Crop the detected hand, resize it to IMG_SIZE x IMG_SIZE,
    and place it on a white background.
    """
    x, y, w, h = hand["bbox"]

    # Keep bounding box inside frame
    y1 = max(0, y - OFFSET)
    y2 = min(img.shape[0], y + h + OFFSET)
    x1 = max(0, x - OFFSET)
    x2 = min(img.shape[1], x + w + OFFSET)

    img_crop = img[y1:y2, x1:x2]

    if img_crop.size == 0:
        return None, None

    img_white = np.ones(
        (IMG_SIZE, IMG_SIZE, 3),
        np.uint8
    ) * 255

    aspect_ratio = h / w

    if aspect_ratio > 1:
        k = IMG_SIZE / h
        w_cal = math.ceil(k * w)

        img_resize = cv2.resize(
            img_crop,
            (w_cal, IMG_SIZE)
        )

        w_gap = math.ceil(
            (IMG_SIZE - w_cal) / 2
        )

        img_white[
            :,
            w_gap:w_gap + w_cal
        ] = img_resize

    else:
        k = IMG_SIZE / w
        h_cal = math.ceil(k * h)

        img_resize = cv2.resize(
            img_crop,
            (IMG_SIZE, h_cal)
        )

        h_gap = math.ceil(
            (IMG_SIZE - h_cal) / 2
        )

        img_white[
            h_gap:h_gap + h_cal,
            :
        ] = img_resize

    return img_crop, img_white, (x1, y1, x2, y2)


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(
        maxHands=1,
        detectionCon=MIN_CONF
    )

    counter = 0
    last_save_time = 0

    print(f"📸 Collecting images for gesture: '{GESTURE_NAME}'")
    print("👉 Press 'S' to save.")
    print("👉 Press 'Q' to quit.")

    while True:
        success, img = cap.read()

        if not success:
            print("❌ Unable to open camera.")
            break

        hands, img = detector.findHands(
            img,
            flipType=True,
            draw=False
        )

        img_white = None

        if hands:
            hand = hands[0]

            result = prepare_image(img, hand)

            if result[0] is not None:
                img_crop, img_white, bbox = result
                x1, y1, x2, y2 = bbox

                # Preview
                cv2.imshow("Image Crop", img_crop)
                cv2.imshow("White Background", img_white)

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 255),
                    2
                )

                cv2.putText(
                    img,
                    f"Ready: {GESTURE_NAME}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 0),
                    3
                )

        cv2.imshow("Main Feed", img)

        key = cv2.waitKey(1) & 0xFF

        # === Save image ===
        if key == ord("s") and hands and img_white is not None:

            if time.time() - last_save_time > SAVE_DELAY:
                counter += 1

                filename = (
                    f"Image_{GESTURE_NAME}_"
                    f"{time.time()}_{counter}.jpg"
                )

                filepath = os.path.join(
                    FOLDER,
                    filename
                )

                cv2.imwrite(filepath, img_white)

                print(f"✅ Saved: {filepath}")

                last_save_time = time.time()

        # === Quit ===
        elif key == ord("q"):
            print("🛑 Dataset collection stopped.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()