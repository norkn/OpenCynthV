import cv2
import numpy as np

from shape import center_of_shape_cam as sh
import connectionDetector as cd
import utils


def mouseCallback(event, x, y, flags, param):

    global frame, maxx, maxy

    if event == cv2.EVENT_LBUTTONDOWN:

        modules_whiteout = utils.getWhiteoutByHue(frame, 0, 100)

        no_connections_mask = cv2.bitwise_and(modules_whiteout, sh._preprocess(frame))
        no_connections_mask_inverted = cv2.bitwise_not(no_connections_mask)
        no_connections_mask_inverted = cv2.merge((no_connections_mask_inverted, no_connections_mask_inverted, no_connections_mask_inverted))
        kernel = np.ones((3, 3), np.uint8)
        no_connections_mask_inverted = cv2.erode(no_connections_mask_inverted, kernel, iterations=3)
        frame_modules_only = cv2.add(frame, no_connections_mask_inverted)

        detected_shapes = sh.findShapes(frame_modules_only, maxx, maxy)
        nodes, edges = cd.traceConnections(frame, detected_shapes, modules_whiteout, 28)

        # cv2.imshow('sh preprocessing on original', sh._preprocess(frame))
        # cv2.imshow('modules_mask', modules_whiteout)
        # cv2.imshow('modules_only ', no_connections_mask)
        # cv2.imshow('module detection', frame_modules_only)
        # cv2.imshow('tracing preprocessing on original', cd._preprocess(frame))

        for s in detected_shapes:

            shape = s[0]
            cX, cY = s[1]
            c = s[2]

            cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(frame, str(shape), (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), )
            cv2.drawContours(frame, [c], -1, (50, 240, 240), 2)
        
        for e in edges:
            p = e[0]
            q = e[1]

            frame = cv2.line(frame, p, q, color=(0, 0, 255), thickness=2)
        
        cv2.imshow('original', frame)

        cv2.waitKey(0)


##################################################
frame = cv2.imread('test_images/connections_test_5.jpg', cv2.IMREAD_COLOR)
maxx, maxy = frame.shape[1], frame.shape[0]
cv2.namedWindow('original')
cv2.imshow('original', frame)
cv2.setMouseCallback('original', mouseCallback)
cv2.waitKey(0)
cv2.destroyAllWindows()
