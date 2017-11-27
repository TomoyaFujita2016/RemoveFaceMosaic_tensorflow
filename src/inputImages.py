import tensorflow as tf
import cv2
import os

def inputImages(imageDir):
    trainImages = []
    imgfiles = os.listdir(imageDir)
    for imgfile in imgfiles:
        img = cv2.imread(imageDir + imgfile)
        # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32)/255.0
        trainImages.append(img)
    return train

def setupInputdata():
    NO_MOSAIC_DIR = "../Dataset/NoMosaicImages/"
    MOSAIC_DIR = "../Dataset/1MosaicImages/"
    
