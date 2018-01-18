import tensorflow as tf
import numpy as np
import cv2
import os
import glob
from tqdm import tqdm

FLAGS = tf.app.flags.FLAGS


def setupInputdata(mosaicImageDir, labelImageDir):
    labelImages = inputImages(labelImageDir)
    mosaicImages = inputImages(mosaicImageDir)
    generateCsv(mosaicImageDir)
    return labelImages, mosaicImages

def generateCsv(dirPath):
    if dirPath[-1] == "/":
        path = dirPath + "*"
    else:
        path = dirPath + "/*"
    
    fileList = glob.glob(path)
    fileName = "".join(list(filter(lambda s:s!="",dirPath.split("/")))[-1]) + ".csv"

    with open(FLAGS.csvDir + fileName, "w") as f:
        for fl in fileList:
            f.write(fl+"\n")

def inputImages(imageDir):
    trainImages = []
    imgfiles = os.listdir(imageDir)
    
    pBar = tqdm(imgfiles, total=len(imgfiles), desc="Load Data from " + imageDir)
    for i, imgfile in enumerate(pBar):
        pBar.update(1)
        img = cv2.imread(imageDir + imgfile)
        # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32)/255.0
        trainImages.append(img)
    imagesNumpy = np.asarray(trainImages)
    #return imagesNumpy
    imageTensors = tf.Variable(imagesNumpy)
    return imageTensors
