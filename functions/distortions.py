import numpy as np
import cv2


def jpeg(img, quality):
    _, x = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(x, cv2.IMREAD_COLOR)


def resize(img, w, h):
    orig_h, orig_w = img.shape[:2]
    x = cv2.resize(img, (w, h))
    return cv2.resize(x, (orig_w, orig_h))


def canny(img):
    x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = cv2.Canny(x, 100, 200)
    cv2.medianBlur
    return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)


def gaussian_noise(img, mean=0, std=25):
    noise = np.random.normal(mean, std, img.shape).astype(np.uint8)
    noisy_img = cv2.add(img, noise)
    return noisy_img


def blur(img, ksize=(7, 7), sigmaX=0):
    return cv2.GaussianBlur(img, ksize, sigmaX)


def grayscale(img):
    x = img.copy()
    (row, col) = x.shape[0:2]

    for i in range(row):
        for j in range(col):
            x[i, j] = sum(x[i, j]) * 0.33

    return x


def negative(img):
    return cv2.bitwise_not(img)


def zoom(img, zoom, coord=None):
    h, w, _ = [zoom * i for i in img.shape]

    if coord is None:
        cx, cy = w / 2, h / 2
    else:
        cx, cy = [zoom * c for c in coord]

    img = cv2.resize(img, (0, 0), fx=zoom, fy=zoom)
    img = img[
        int(round(cy - h / zoom * 0.5)) : int(round(cy + h / zoom * 0.5)),
        int(round(cx - w / zoom * 0.5)) : int(round(cx + w / zoom * 0.5)),
        :,
    ]

    return img
