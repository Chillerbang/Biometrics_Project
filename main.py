import random
import sys
import glob
import numpy as np
import tkinter.messagebox
import argparse
import time
import os
import cv2 as oc
from tkinter import *
from PIL import Image, ImageTk
import Update_Model

videocapture = oc.VideoCapture(0)
fishfaceClassifer = oc.face.FisherFaceRecognizer_create()
try:
    fishfaceClassifer.read("trained_emoclassifier.xml")
except:
    print("No classifier xml made")
parser = argparse.ArgumentParser(description="Options for emotion learning")
parser.add_argument("--update", help = "", action="store_true")
args = parser.parse_args()
#emotionDictonary so more than 1 face id used to get an accurate emotoin
faceDictionary = {}
emotions = ["angry","happy", "sad", "neutral"]

trainFrontFace = oc.CascadeClassifier("haarcascade_frontalface_default.xml")
def stopcall():
   print("well this broke?")

def cropFace(clache_image, face):
    for (x, y, w, h) in face:
        faceOnly = clache_image[y:y+h, x:x+w]
        faceOnly = oc.resize(faceOnly, (350, 350))
    faceDictionary["face%s" %(len(faceDictionary)+1)] = faceOnly
    return faceOnly

def checkFolder(emotions):
    for e in emotions:
        if os.path.exists("dataset\\%s" %e):
            pass
        else:
            os.makedirs("dataset\\%s" %e)

def detectFace():
    frame = getFrame()
    gray = oc.cvtColor(frame, oc.COLOR_BGR2GRAY)
    clahe = oc.createCLAHE(clipLimit=1.0, tileGridSize=(10, 10))
    clahe_image = clahe.apply(gray)
    face = trainFrontFace.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=20, minSize=(10, 10), flags=oc.CASCADE_SCALE_IMAGE)
    if len(face) != 1:
        print("Get that other person out of the frame or get your face in")
    else:
        print("One Face Detected")
        faceOnly = cropFace(clahe_image, face)
        oc.imshow("detectedFace", faceOnly)
        return faceOnly

def recogniseEmotion():
    prediction = []
    confidence = []
    for f in faceDictionary.keys():
        predFish, conFish = fishfaceClassifer.predict(faceDictionary[f])
        oc.imwrite("images\\%s.jpg" %f, faceDictionary[f])
        prediction.append(predFish)
        confidence.append(conFish)
    print("I think You are : %s" % emotions[max(set(prediction), key=prediction.count)])
    return "Detected emotion: %s" %emotions[max(set(prediction), key=prediction.count)]

def getFrame():
    ret, frame = videocapture.read()
    frame = oc.flip(frame, 1)
    return frame
    #for (x, y, w, h) in face:
    #    oc.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    #cameraframe = oc.cvtColor(frame, oc.COLOR_BGR2RGBA)

def saveFace(emotion):
    print("\n\n please look into the camera and be like " + emotion + " until processing is done, about 5 seconds")
    for i in range(0,5):
        print(5 - i)
        time.sleep(1)
    while len(faceDictionary.keys()) < 16:
        detectFace()
    for k in faceDictionary.keys():
        oc.imwrite("dataset\\%s\\%s.jpg" %(emotion, len(glob.glob("dataset\\%s*" %emotion))), faceDictionary[k])
    faceDictionary.clear()

def updateModel(emotions):
    checkFolder(emotions)
    for i in range(0, len(emotions)):
        saveFace(emotions[i])
    print("images collected")
    Update_Model.update(emotions)
    print("did it break? i dont know i need some sleep")

#def createImage():
##creatin of gui
GUI = Tk()
label = Label(GUI)
label.pack(fill=BOTH, expand=True)
labelSongCurrent = Label(GUI, padx=10, text="Current song:")
labelSongCurrent.pack(side=RIGHT)
labelSongNext = Label(GUI, padx=10, text="Recommended song:")
labelSongNext.pack(side=RIGHT)
labelEmotion = Label(GUI, padx=10, text="Detected emotion:")
labelEmotion.pack(side=RIGHT, padx=10)
buttonStart = Button(GUI, text="Start", command=stopcall, width=20, height=3)
buttonStart.pack(side=LEFT)

buttonStop = Button(GUI, text="Stop Analysis", command=stopcall, width=20, height=3)
buttonStop.pack(side=LEFT, padx=5, pady=5)
"""#enable this code to train
while True:
    #while len(faceDictionary) != 10:
        #detectFace()
    #recogniseEmotion() fisher breaks if you dont update it first
    if args.update:
        updateModel(emotions)
        break
#enable this code to play with the app
"""
def generatecamerafeed():
    detectFace()
    #if args.update:
    #updateModel(emotions)
    #exit(1)
#elif
    if len(faceDictionary) == 10:
        labelEmotion.config(text=recogniseEmotion())
        faceDictionary.clear()
    display = getFrame()
    gray = oc.cvtColor(display, oc.COLOR_BGR2GRAY)
    face = trainFrontFace.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(10, 10), flags=oc.CASCADE_SCALE_IMAGE)
    #if len(faceDictionary) == 5:
        #break
       # print("Yeet An emotion collected")
    for (x, y, w, h) in face:
       oc.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cameraframe = oc.cvtColor(display, oc.COLOR_BGR2RGBA)
    img = Image.fromarray(cameraframe)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, generatecamerafeed)
generatecamerafeed()

GUI.mainloop()
