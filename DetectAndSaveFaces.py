import cv2
import numpy as np
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
id=input('enter user Id')
sampleNo=0
cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.32, 5)
    for (x,y,w,h) in faces:
        sampleNo=sampleNo+1
        cv2.imwrite('Dataset/vishnu/User.'+str(id)+'.'+str(sampleNo)+'.JPG',gray[y:y+h, x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.waitKey(100)
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if sampleNo>20:
        break
cap.release()
cv2.destroyAllWindows()