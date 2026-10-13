import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

# === Configuration ===
OFFSET = 30
IMG_SIZE = 640
MIN_CONF = 0.7
# =======================


def prepare_image(img, hand):
    x, y, w, h = hand["bbox"]

    y1 = max(0, y - OFFSET)
    y2 = min(img.shape[0], y + h + OFFSET)
    x1 = max(0, x - OFFSET)
    x2 = min(img.shape[1], x + w + OFFSET)

    img_crop = img[y1:y2, x1:x2]

    if img_crop.size == 0:
        return None

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

    return img_white


def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(
        maxHands=1,
        detectionCon=MIN_CONF
    )

    print("🧪 Gesture test mode")
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

        if hands:
            hand = hands[0]

            img_white = prepare_image(img, hand)

            if img_white is not None:
                cv2.imshow(
                    "Processed Image",
                    img_white
                )

        cv2.imshow("Main Feed", img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()