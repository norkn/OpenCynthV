import cv2
import numpy as np


_INF = 0xFFFFFFFF


def getWhiteoutByHue(img, hue, threshold):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h, s, v =  cv2.split(img)
    
    h_whiteout = cv2.inRange(h, hue - threshold, hue + threshold)

    lower_hsv = np.array([hue - threshold,0,0])
    upper_hsv = np.array([hue + threshold,255,255])
    
    h_whiteout = cv2.inRange(img, lower_hsv, upper_hsv)

    return h_whiteout


def indexOfClosestElementInList(e, l):

    min_distance = _INF
    index = -1

    for i in range(len(l)):

        d = np.linalg.norm(l[i] - e)

        if d < min_distance:
            min_distance = d
            index = i

    return index


def countVotes(l):

    dictionary = {}

    for element in l:

        element = tuple(element)

        if element in dictionary:
            dictionary[element] += 1
        else:
            dictionary[element] = 1

    result = None
    max_votes = -1

    for element in dictionary.keys():

        if dictionary[element] > max_votes:
            result = element
            max_votes = dictionary[element]

    return result