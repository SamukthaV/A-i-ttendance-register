import os,cv2
import numpy as np
from PIL import Image
recognizer = cv2.face.LBPHFaceRecognizer_create()
path1='Dataset/sab'
path2='Dataset/vishnu'
path3 = 'Dataset/samu'
path5 = 'Dataset/bhanu'
def getImagePath1(path):
    imagePath=[os.path.join(path,f) for f in os.listdir(path)]
    print(imagePath)
    faces=[]
    IDs=[]
    for image in imagePath:
        img=Image.open(image).convert('L')
        faceNP=np.array(img,'uint8')
        faces.append(faceNP)
        IDs.append(1)
        #cv2.imshow('training',faceNP)
        #cv2.waitKey(20)
    return np.array(IDs),faces

def getImagePath2(path):
    imagePath=[os.path.join(path,f) for f in os.listdir(path)]
    print(imagePath)
    faces=[]
    IDs=[]
    for image in imagePath:
        img=Image.open(image).convert('L')
        faceNP=np.array(img,'uint8')
        faces.append(faceNP)
        IDs.append(2)
        #cv2.imshow('training',faceNP)
        #cv2.waitKey(20)
    return np.array(IDs),faces
def getImagePath3(path):
    imagePath=[os.path.join(path,f) for f in os.listdir(path)]
    print(imagePath)
    faces=[]
    IDs=[]
    for image in imagePath:
        img=Image.open(image).convert('L')
        faceNP=np.array(img,'uint8')
        faces.append(faceNP)
        IDs.append(3)
        cv2.imshow('training',faceNP)
        cv2.waitKey(20)
    return np.array(IDs),faces
def getImagePath5(path):
    imagePath=[os.path.join(path,f) for f in os.listdir(path)]
    print(imagePath)
    faces=[]
    IDs=[]
    for image in imagePath:
        img=Image.open(image).convert('L')
        faceNP=np.array(img,'uint8')
        faces.append(faceNP)
        IDs.append(5)
        #cv2.imshow('training',faceNP)
        #cv2.waitKey(20)
    return np.array(IDs),faces
IDs1,faces1=getImagePath1(path1)
IDs2,faces2=getImagePath2(path2)
IDs3,faces3=getImagePath3(path3)
IDs5,faces5=getImagePath5(path5)
recognizer.train(np.concatenate((faces1,faces2,faces3,faces5)),np.concatenate((IDs1,IDs2,IDs3,IDs5)))
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()