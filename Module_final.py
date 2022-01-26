# Quelle: https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
# Ergänzende Quelle wg division by zero error (aus Skript): https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/ 
#https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html (wg threshold anpassung)
# weitere: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
#https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html

#https://dev.to/simarpreetsingh019/detecting-geometrical-shapes-in-an-image-using-opencv-4g72
# https://docs.opencv.org/3.4/d1/d32/tutorial_py_contour_properties.html

# import the necessary packages

import imutils
import cv2


def _preprocess(img):

    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,149,4)
    opening= cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)

    invert = cv2.bitwise_not(opening)

    return invert

    
def _identify(c):

    # initialize the shape name and approximate the contour
    shape = "unidentified"
    epsilon = 0.03*cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,epsilon,True)

    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "WaveShaping"
    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "Convolution" if ar >= 0.90 and ar <= 1.10 else "Oszillator"
        # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "StereoPanner"
    # otherwise, we assume the shape is a circle
    else:
        shape = "In/Out"

    return shape


def _getCenterOfShape(c):
    
    # compute the center of the contour
    M = cv2.moments(c)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    return cX, cY


def _select(cnts, maxx, maxy, min_area):

    detectedShapes = []
    detectedContours = []

    # loop over the contours
    for c in cnts:

        area=cv2.contourArea(c)
        
        if area / (maxx * maxy) > min_area:

            leftmost = tuple(c[c[:,:,0].argmin()][0])
            rightmost = tuple(c[c[:,:,0].argmax()][0])
            topmost = tuple(c[c[:,:,1].argmin()][0])
            bottommost = tuple(c[c[:,:,1].argmax()][0])
            
            if leftmost[0] > 0 and rightmost[0] < (maxx-1) and topmost[1] > 0 and bottommost[1] < (maxy-1):
                
                shape = _identify(c)
                if shape == 'In/Out':
                    leftmost =  None
                    rightmost = None
                    topmost = None
                if shape != 'Oszillator':
                    topmost = None
              
                (cX, cY) = _getCenterOfShape(c)
                detectedShapes.append((shape, (cX, cY), (leftmost, rightmost, topmost)))
                detectedContours.append(c)

    return detectedShapes, detectedContours


def findShapes(img, maxx, maxy, min_area):

    img = _preprocess(img)

    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    detectedShapes, detectedContours = _select(cnts, maxx, maxy, min_area)
    return detectedShapes, detectedContours
