import cv2

from shape import center_of_shape_cam as sh
import connectionDetector as cd


def mouseCallback(event, x, y, flags, param):

    global frame, maxx, maxy

    if event == cv2.EVENT_LBUTTONDOWN:
        detected_shapes = sh.findShapes(frame, maxx, maxy)
        nodes, edges = cd.traceConnections(frame, detected_shapes, 28)
        frame = cv2.cvtColor(sh._preprocess(frame), cv2.COLOR_GRAY2BGR)

        for s in detected_shapes:

            shape = s[0]
            cX, cY = s[1]
            c = s[2]

            cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(frame, str(shape), (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), )
            cv2.drawContours(frame, [c], -1, (50, 240, 240), 2)
        
        cv2.imshow('Connection Detection', frame)

        cv2.waitKey(0)


video = cv2.VideoCapture(0)

maxx = video.get(cv2.CAP_PROP_FRAME_WIDTH)
maxy = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

fps = video.get(cv2.CAP_PROP_FPS)

cv2.namedWindow('Connection Detection')
cv2.setMouseCallback('Connection Detection', mouseCallback)

while video.isOpened():
    
    ret, frame = video.read()
    
    cv2.imshow('Connection Detection', frame)
    
    if cv2.waitKey(int(1000 / fps) + 1) != -1:
        break

video.release()
cv2.destroyAllWindows()
