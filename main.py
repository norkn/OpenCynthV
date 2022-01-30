import time

import cv2

import imageToGraph as im2G


def _drawTrace(frame, graph):

    if graph is not None:

        for node in graph:

            p = node
            q = graph[node]

            frame = cv2.line(frame, p[1], q[1], color=(0, 0, 255), thickness=2)


def _drawShapes(frame, detected_shapes, detected_contours):

    if detected_contours is not None:

        for c in detected_contours:
            
            cv2.drawContours(frame, [c], -1, (50, 240, 240), 2)

    if detected_shapes is not None:

        for s in detected_shapes:

            shape = s[0]
            cX, cY = s[1]

            cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(frame, str(shape), (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), )

            color = [(120,130,90), (80, 120, 100), (110, 90, 120)]
            for i in range(len(s[2])):
                point = s[2][i]
                if point is not None:
                    cv2.circle(frame, (point[0], point[1]), 7, color[i], -1)


def hold(frame):
    global update_shapes_continuously
    global params, graph
    global shapes, shape_contours, last_state_was_connected
    
    shapes, shape_contours, last_state_was_connected = im2G.registerModules(frame, params[0], params[1], params[2], graph)
    update_shapes_continuously = False

    return


def updateGraph(frame, params):

    global last_time, update_shapes_continuously
    global shapes, shape_contours, last_state_was_connected
    global graph

    if update_shapes_continuously:
        shapes, shape_contours, last_state_was_connected = im2G.registerModules(frame, params[0], params[1], params[2], graph)

    t = time.time()

    if t - last_time > 0.2:
        graph = im2G.updateConnections(frame, params[0], params[1], shapes, last_state_was_connected, int(params[3]), params[4], graph)
        last_time = t

    return graph


def drawContoursAndConnections(frame):

    global shapes, shape_contours, graph

    shapes_in = shapes.copy()
    shape_contours_in = shape_contours.copy()
    graph_in = graph.copy()

    frame_out = frame.copy()

    _drawShapes(frame_out, shapes_in, shape_contours_in)
    _drawTrace(frame_out, graph_in)

    return frame_out


####################GLOBALS#######################
graph = {}
shapes = []
shape_contours = []
new_edges = None
last_state_was_connected = []

r = 28
modules_whiteout = None

last_time = 0

hue = 0
hue_thresh = 100
min_area = 0.008
r = 28
endpoint_thresh = 30

paramNames = ['hue', 'hue thresh', 'min contour area', 'r', 'endpoint vicinity']
params = [hue, hue_thresh, min_area, r, endpoint_thresh]
paramRanges = [(0, 360), (0, 360), (0, 0.05), (0, 100), (0, 100)]

update_shapes_continuously = True

# ui = UI.UI(paramNames, paramRanges, params)
# ##################################################

# def mouseCallback(event, x, y, flags, param):

#     global graph
#     global frame, r
#     global shapes, shape_contours, last_state_was_connected
#     global frame_to_register, modules_whiteout


#     if event == cv2.EVENT_LBUTTONDOWN:

#         shapes, shape_contours, last_state_was_connected = im2G.registerModules(frame, params[0], params[1], params[2], graph)


# video = cv2.VideoCapture(0)

# maxx = video.get(cv2.CAP_PROP_FRAME_WIDTH)
# maxy = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

# fps = video.get(cv2.CAP_PROP_FPS)


# cv2.namedWindow('original')
# cv2.setMouseCallback('original', mouseCallback)

# while video.isOpened():

#     ret, frame = video.read()

    # params[3] = int(params[3])
    # update(frame, params)

    # wo = im2G._getWhiteoutByHue(frame, params[0], params[1])
    # #mods_preproc = sh._preprocess(frame)
    # only_mods = im2G._applyWhiteout(wo, frame)
    # only_cons = cd._preprocess(frame, wo, int(params[3]))
    # cv2.imshow('only mods', only_mods)
    # cv2.imshow('only_cons', only_cons)

    # if cv2.waitKey(int(1000 / fps) + 1) != -1:
    #     break

# video.release()
# cv2.destroyAllWindows()