import time

import cv2

import imageToGraph as im2G
import UI


def drawTrace(frame_out, graph):

    if graph is not None:

        for node in graph:

            p = node
            q = graph[node]

            frame_out = cv2.line(frame_out, p, q, color=(0, 0, 255), thickness=2)


def drawShapes(frame_out, detected_shapes, detected_contours):

    if detected_contours is not None:

        for c in detected_contours:
            
            cv2.drawContours(frame_out, [c], -1, (50, 240, 240), 2)

    if detected_shapes is not None:

        for s in detected_shapes:

            shape = s[0]
            cX, cY = s[1]

            cv2.circle(frame_out, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(frame_out, str(shape), (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), )


def update(frame, params):

    global last_time
    global shapes, last_state_was_connected
    global new_edges, graph
    global modules_whiteout, frame_to_register

    t = time.time()

    if t - last_time > 0.2:

        graph = im2G.updateConnections(frame, params[0], params[1], shapes, last_state_was_connected, params[3], params[4], graph)
        
        last_time = t


    frame_out = frame.copy()

    drawShapes(frame_out, shapes, shape_contours)
    drawTrace(frame_out, graph)

    cv2.imshow('original', frame_out)


####################GLOBALS#######################
graph = {}
shapes = []
shape_contours = []
new_edges = None
last_state_was_connected = []

r = 28
modules_whiteout = None

frame = None
frame_to_register = None

last_time = 0


hue = 0
hue_thresh = 100
min_area = 0.008
r = 28
endpoint_thresh = 30

paramNames = ['hue', 'hue thresh', 'min contour area', 'r', 'endpoint vicinity']
params = [hue, hue_thresh, min_area, r, endpoint_thresh]
paramRanges = [(0, 360), (0, 360), (0, 0.05), (0, 100), (0, 100)]

ui = UI.UI(paramNames, paramRanges, params)
##################################################

def mouseCallback(event, x, y, flags, param):

    global graph
    global frame, r
    global shapes, shape_contours, last_state_was_connected
    global frame_to_register, modules_whiteout

    if event == cv2.EVENT_LBUTTONDOWN:

        shapes, shape_contours, last_state_was_connected = im2G.registerModules(frame, params[0], params[1], params[2], graph)


video = cv2.VideoCapture(0)

maxx = video.get(cv2.CAP_PROP_FRAME_WIDTH)
maxy = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

fps = video.get(cv2.CAP_PROP_FPS)


cv2.namedWindow('original')
cv2.setMouseCallback('original', mouseCallback)

while video.isOpened():

    ret, frame = video.read()

    update(frame, params)

    if cv2.waitKey(int(1000 / fps) + 1) != -1:
        break

video.release()
cv2.destroyAllWindows()