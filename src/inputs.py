import tensorflow as tf
import numpy as np
import cv2
import os
from tqdm import tqdm

FLAGS = tf.app.flags.FLAGS

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
    imageTensors = tf.Variable(imagesNumpy)
    return imageTensors

def setupInputdata(mosaicImageDir, labelImageDir):
    labelImages = inputImages(labelImageDir)
    mosaicImages = inputImages(mosaicImageDir)

    return labelImages, mosaicImages
