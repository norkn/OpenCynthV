import cv2
import numpy as np


def getWhiteoutByHue(img, hue, threshold):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h, s, v =  cv2.split(img)
    
    h_whiteout = cv2.inRange(h, hue - threshold, hue + threshold)

    lower_hsv = np.array([hue - threshold,0,0])
    upper_hsv = np.array([hue + threshold,255,255])
    
    h_whiteout = cv2.inRange(img, lower_hsv, upper_hsv)

    return h_whiteout