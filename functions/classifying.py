from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2
from skimage.metrics import structural_similarity


MODEL = ResNet50(weights="imagenet")


def classify(img):
    try:
        x = cv2.resize(img, (224, 224))
        x = x[:, :, ::-1].astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = MODEL.predict(x)
        classes = decode_predictions(preds)[0]
        for c in classes:
            print("\t%s (%s): %.2f%%" % (c[1], c[0], c[2] * 100))

    except Exception as e:
        print("Classification failed.")

def classify_to_data(img):
    classify = []
    try:
        x = cv2.resize(img, (224, 224))
        x = x[:, :, ::-1].astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = MODEL.predict(x)
        classes = decode_predictions(preds)[0]
        for c in classes:
            classify.append([c[1], c[0], c[2] * 100])

    except Exception as e:
        print("Classification failed.")

    return classify


def open_img(path):
    return cv2.imread(path)


def ssim(img1, img2):
    return structural_similarity(img1, img2, channel_axis=2) * 100
