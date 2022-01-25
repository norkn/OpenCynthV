import cv2
import numpy as np

from shape import center_of_shape_cam as sh
import connectionDetector as cd
import utils

import UI


def update():

    global frame, maxx, maxy
    global params

    cv2.imshow('original', frame)

    modules_whiteout = utils.getWhiteoutByHue(frame, params[0], params[1])
    no_connections_mask = cv2.bitwise_and(modules_whiteout, sh._preprocess(frame))
    no_connections_mask_inverted = cv2.bitwise_not(no_connections_mask)
    kernel = np.ones((3, 3), np.uint8)
    no_connections_mask_inverted = cv2.erode(no_connections_mask_inverted, kernel, iterations=3)
    no_connections_mask_inverted = cv2.merge((no_connections_mask_inverted, no_connections_mask_inverted, no_connections_mask_inverted))
    frame_modules_only = cv2.bitwise_or(frame, no_connections_mask_inverted)

    detected_shapes = sh.findShapes(frame_modules_only, maxx, maxy, params[2])
    nodes, edges = cd.traceConnections(frame, detected_shapes, modules_whiteout, params[3], params[4])

    #cv2.imshow('sh preprocessing on original', sh._preprocess(frame))
    #cv2.imshow('modules_mask', modules_whiteout)
    #cv2.imshow('modules_only ', no_connections_mask)
    cv2.imshow('module detection', frame_modules_only)
    #cv2.imshow('tracing preprocessing on original', cd._preprocess(frame))

    frame_out = frame.copy()

    for s in detected_shapes:

        shape = s[0]
        cX, cY = s[1]
        c = s[2]

        cv2.circle(frame_out, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(frame_out, str(shape), (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), )
        cv2.drawContours(frame_out, [c], -1, (50, 240, 240), 2)
    
    for e in edges:

        p = e[0]
        q = e[1]

        frame_out = cv2.line(frame_out, p, q, color=(0, 0, 255), thickness=2)
    
    cv2.imshow('original', frame_out)

    #cv2.waitKey(0)


##################################################
frame = cv2.imread('test_images/connections_test_5.jpg', cv2.IMREAD_COLOR)
maxx, maxy = frame.shape[1], frame.shape[0]

hue = 0
hue_thresh = 100
min_area = 0.008
r = 28
endpoint_thresh = 30

paramNames = ['hue', 'hue thresh', 'min contour area', 'r', 'endpoint vicinity']
params = [hue, hue_thresh, min_area, r, endpoint_thresh]
paramRanges = [(0, 360), (0, 360), (0, 0.05), (0, 100), (0, 100)]

ui = UI.UI(paramNames, paramRanges, params)

cv2.namedWindow('original')
# cv2.setMouseCallback('original', update)

while True:

    update()

    if cv2.waitKey(int(1000 / 0.5) + 1) != -1:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
