from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from multiprocessing import Process
import datetime
from playsound import playsound
import matplotlib.pyplot as plt

def webcam_video():
    proto_txt_path = 'deploy.prototxt'
    model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
    face_detector = cv2.dnn.readNetFromCaffe(proto_txt_path, model_path)

    mask_detector = load_model('mask_detector.model')
    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        ret, frame = cap.read()
        print(ret)
        print(frame)
    else:
        ret = False

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #plt.imshow(image)
    #plt.title('Color Image RGB')
    #plt.xticks([])
    #plt.yticks([])
    #plt.show()

    cap.release()

    #image = cv2.imread("tony.jpg")

    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 177, 123))

    face_detector.setInput(blob)
    detections = face_detector.forward()

    faces = []
    bbox = []
    results = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            faces.append(face)
            bbox.append((startX, startY, endX, endY))

    if len(faces) > 0:
        results = mask_detector.predict(faces)

    for (face_box, result) in zip(bbox, results):
        (startX, startY, endX, endY) = face_box
        (mask, withoutMask) = result

        label = ""
        if mask > withoutMask:
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)
            t=Process(target=local_video)
            t.start()
        cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", image)
    cv2.waitKey(0)


def local_video():
    playsound("beep-11.wav")


if __name__ == '__main__':
    p2 = Process(target=webcam_video)
    p2.start()

    p2.join()
