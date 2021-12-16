import cv2
import numpy as np


vid = cv2.VideoCapture('VIDEO2.mp4')
frameCount = 0
name = ''

while vid.isOpened():

    ret, frame = vid.read()
    
    if frameCount == 170:
        midFrame =frame.copy()  

        name = './midframe' + str(frameCount) + '.jpg'
        print ('Creating...' + name)
  
        # writing the extracted images
        cv2.imwrite(name, midFrame)
        cv2.imshow('vid', midFrame)
    frameCount += 1

    
    if cv2.waitKey(25) != -1:
        break;

vid.release()
cv2.destroyAllWindows()

def mouseCallback(event, x, y, flags, param):
    new_img = img.copy()
    (b, g, r) = img[y, x]
    col = (int(b), int(g), int(r))
    cv2.rectangle(new_img, (0, 0), (30, 30), col, cv2.FILLED)
    cv2.putText(new_img, str((r, g, b)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow('image', new_img)

img = cv2.imread(name, cv2.IMREAD_COLOR)
cv2.namedWindow('image')
cv2.setMouseCallback('image', mouseCallback)
cv2.waitKey(0)

cv2.destroyAllWindows()
