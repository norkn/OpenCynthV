# Quelle: https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
# Ergänzende Quelle wg division by zero error (aus Skript): https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/ 
#https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html (wg threshold anpassung)
# weitere: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
#https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html

# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np


#cap = cv2.VideoCapture('VIDEO1.mp4')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())
vid = cv2.VideoCapture(args["image"])

while vid.isOpened():
	ret, cap = vid.read()

	# load the image, convert it to grayscale, blur it slightly,
	# and threshold it
	#image = cv2.imread(args["image"])
	gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,149,4)
	invert = cv2.bitwise_not(thresh)
	#thresh = cv2.threshold(blurred, 105, 255, cv2.THRESH_BINARY)[1]

	# find contours in the thresholded image
	cnts = cv2.findContours(invert.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# loop over the contours
	for c in cnts:
		# compute the center of the contour
		M = cv2.moments(c)
		if M["m00"] != 0:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
		else:
			cX, cY = 0, 0

		#cX = int(M["m10"] / M["m00"])
		#cY = int(M["m01"] / M["m00"])

		# draw the contour and center of the shape on the image
		cv2.drawContours(cap, [c], -1, (0, 255, 0), 2)
		cv2.circle(cap, (cX, cY), 7, (255, 255, 255), -1)
		cv2.putText(cap, "center", (cX - 20, cY - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			
		# show the image
		cv2.imshow("cap", cap)
	if cv2.waitKey(50) != -1:
		break;

vid.release()
cv2.destroyAllWindows()
