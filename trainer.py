import os,cv2
import numpy as np
from PIL import Image
recognizer = cv2.face.LBPHFaceRecognizer_create()
path1='Dataset/sab'
path2='Dataset/vishnu'
def getImagePath1(path):
    imagePath=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for image in imagePath:
        img=Image.open(image).convert('L')
        faceNP=np.array(img,'uint8')
        faces.append(faceNP)
        IDs.append(1)
        cv2.imshow('training',faceNP)
        cv2.waitKey(20)
    return np.array(IDs),faces

def getImagePath2(path):
    imagePath=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for image in imagePath:
        img=Image.open(image).convert('L')
        faceNP=np.array(img,'uint8')
        faces.append(faceNP)
        IDs.append(2)
        cv2.imshow('training',faceNP)
        cv2.waitKey(20)
    return np.array(IDs),faces

IDs1,faces1=getImagePath1(path1)
IDs2,faces2=getImagePath2(path2)
recognizer.train(np.concatenate((faces2,faces1)),np.concatenate((IDs1,IDs2)))
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()