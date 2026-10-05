import cv2
import numpy as np

def apply_clahe(img):
    # Expected input: (224, 224, 1), float32 in [0, 1]
    img = np.squeeze(img)
    img = np.uint8(img * 255)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    img = clahe.apply(img)

    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=-1)
