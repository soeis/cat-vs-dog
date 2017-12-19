# -*- coding=utf-8 -*-

import cv2
import os

cwd = os.getcwd()
# catPath = cwd + "\haarcascade_frontalcatface_extended.xml"
catPath = cwd + "\haarcascade_frontalcatface.xml"
faceCascade = cv2.CascadeClassifier(catPath)

# read img, and 2gray
img = cv2.imread("train/cat.1.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cat face detect
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.02,
    minNeighbors=3,
    minSize=(100, 100),
    flags=cv2.CASCADE_SCALE_IMAGE
)

# quote cat face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(img, 'Cat', (x, y - 7), 3, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

# show img
cv2.imshow('Cat?', img)
# cv2.imwrite("cat.jpg", img)
c = cv2.waitKey(0)
