import cv2

import connectionDetector as cd

#########################################


def mouseCallback(event, x, y, flags, param):

    #detect shapes in current frame, get coordinates and shapetype

    #give (coord, type) list to cd, call a trace

    global input_img
    if event == cv2.EVENT_LBUTTONDOWN:
        cd._traceConnection(input_img, x, y, 25)
    
    
############################
input_img = cv2.imread('connections_test_2.jpg', cv2.IMREAD_COLOR)
cv2.namedWindow('Connection Detection')
cv2.imshow('Connection Detection', input_img)
cv2.setMouseCallback('Connection Detection', mouseCallback)
cv2.waitKey(0)
cv2.destroyAllWindows()
############################

# video = cv2.VideoCapture(0)
# ret, img = video.read()

# fps = video.get(cv2.CAP_PROP_FPS)

# cv2.namedWindow('Connection Detection')
# cv2.setMouseCallback('Connection Detection', mouseCallback)

# isRunning = False

# while video.isOpened():
    
#     if not isRunning:
#         ret, frame = video.read()
    
#         cv2.imshow('Connection Detection', frame)
    
#     if cv2.waitKey(int(1000 / fps) + 1) != -1:
#         break

#cv2.destroyAllWindows()
