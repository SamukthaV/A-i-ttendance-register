import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer/trainingData.yml')
id=0
font = cv2.FONT_HERSHEY_SIMPLEX
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.05, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        id,conf=rec.predict(gray[y:y+h, x:x+w])
        print(id)
        if id==1:
            id='sab'
        if id==2:
            id='vishnu'
        if id==3:
            id='samu'
        if id==5:
            id='bhanu'
        cv2.putText(img,str(id),(x,y+h),font,1, (116, 90, 53), 2, cv2.LINE_AA)
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()