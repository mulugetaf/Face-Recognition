import os
import pickle
import shutil
import tkinter as tk
# GUI
from tkinter import *
from tkinter import filedialog, ttk

import cv2
import face_recognition
import numpy as np
from imutils import paths
from PIL import Image, ImageTk
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
    
def train_facerecogntion(self):
    
        #get paths of each file in folder named Images
        #Images here contains my data(folders of various persons)
        imagePaths = list(paths.list_images(r"DATASET1"))
        knownEncodings = []
        knownNames = []
            # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            name = imagePath.split(os.path.sep)[-2]
            # load the input image and convert it from BGR (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #Use Face_recognition to locate faces
            boxes = face_recognition.face_locations(rgb,model='hog')
            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)
            # loop over the encodings
            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)
            #save emcodings along with their names in dictionary data
        data = {"encodings": knownEncodings, "names": knownNames}
            #use pickle to save data into a file for later use
        f = open("face_enc", "wb")
        f.write(pickle.dumps(data))
        f.close() 
def creat_directory(self, name:str, parent_dir:str):
    
        # Directory
        dir_name =  name

        # Parent Directory path or current dir path

        # Path
        path = os.path.join(parent_dir, dir_name)
        os.mkdir(path)
        