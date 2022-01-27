import time
import math

import numpy as np
import cv2

import utils


def DEBUG_VISUAL(slopes, img, r, current_point, last_slope):

    new_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.rectangle(new_img, tuple(current_point - (r, r)),
                  tuple(current_point + (r, r)), (0, 255, 0))

    cv2.circle(new_img, tuple(current_point), 2, (0, 0, 255), thickness=-1)

    for s in slopes:
        cv2.circle(new_img, tuple(current_point + s),
                   2, (0, 0, 255), thickness=-1)

    cv2.circle(new_img, tuple(current_point + last_slope),
               3, (255, 100, 90), thickness=-1)

    cv2.imshow('Connection Detection', new_img)
    if cv2.waitKey(10) != -1:
        if cv2.waitKey(10) != -1:
            cv2.destroyAllWindows()
            quit()


_endpoint_threshold = 30
_endpoints = []

whiteout = None

_TIMEOUT = 1

def _isPointsClose(p, q):
    print("points", p, q, "distance:", np.linalg.norm(np.array(p) - np.array(q)))
    return np.linalg.norm(np.array(p) - np.array(q)) < _endpoint_threshold

def _isShapesClose(shape_a, shape_b):
    p = shape_a[1]
    q = shape_b[1]
    return _isPointsClose(p, q)

def _endpointCloseTo(p):
    for e in _endpoints:
        if _isPointsClose(p, e[1]): #_isShapesClose(p, e):
            return e
    return None

def _isCloseToAnEndpoint(p):
    print("num endpoints:",len(_endpoints))
    for e in _endpoints:
        print(e, p)
        if _isShapesClose(p, e):
            return True
    return False


def _isCloseToBlack(c):

    threshold = 0.5
    color_magnitude = c #math.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2)
    return color_magnitude < threshold


  
def _getPerimeterBorder(r):
    perimeter_left   = [np.array([0        , 2*r-1 - i]) for i in range(2*r)]
    perimeter_top    = [np.array([i        , 0        ]) for i in range(2*r)]
    perimeter_right  = [np.array([2*r-1    , i        ]) for i in range(2*r)]    
    perimeter_bottom = [np.array([2*r-1 - i, 2*r-1    ]) for i in range(2*r)]
    
    return perimeter_left + perimeter_top + perimeter_right + perimeter_bottom


def _collectPotentialNextPoints(img, r, x, y):

    perimeter_border = _getPerimeterBorder(r)

    intersections = []

    isOnIntersection = False
    avg_index = 0
    i = 0
    count = 0

    for p in perimeter_border:

        i += 1

        if _isCloseToBlack(img[y - r + p[1], x - r + p[0]]):
            isOnIntersection = True
            avg_index += i
            count += 1            
        else:
            if isOnIntersection:
                isOnIntersection = False
                avg_index = int(avg_index / count)
                intersections.append(
                    perimeter_border[avg_index] - np.array([r, r]))
                avg_index = 0
                count = 0

    return intersections


def _preprocess(img, whiteout, r):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((3, 3), np.uint8)

    img = cv2.GaussianBlur(img, (3, 3), 8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.medianBlur(img, 5)
    img = cv2.dilate(img, kernel, iterations=3)
    
    img = cv2.add(img, whiteout)

    img = cv2.erode(img, kernel, iterations=2)

    padded = np.ones((r + img.shape[0], r + img.shape[1]), np.uint8)
    padded[r : r + img.shape[0], r : r + img.shape[1]] = img[:,:]
    img = padded

    return img


def _closestSlope(slope, slopes):

    slope_magnitude = np.linalg.norm(slope)
    normalized_slope = slope / slope_magnitude

    max_dot_product = -1
    best_slope = np.zeros(shape=(2,), dtype=np.int32)

    for s in slopes:

        dot_product = sum(normalized_slope * s / np.linalg.norm(s))

        if (dot_product > max_dot_product):

            max_dot_product = dot_product
            best_slope = s

    return best_slope


def _colorInjection(last_slope, slopes, window):

    window = window.copy()

    start_index = utils.indexOfClosestElementInList(_closestSlope(
        last_slope, slopes), slopes)  # indexOfClosestElementInList(last_slope, slopes)
    r = int(window.shape[0] / 2)
    perimeter_border = _getPerimeterBorder(r)

    injection_points = []

    #inject color
    for i in range(len(slopes)):

        index = (start_index + i) % len(slopes)

        injection_point = slopes[index] + (r, r)
        border_index = utils.indexOfClosestElementInList(
            injection_point, perimeter_border)

        while _isCloseToBlack(window[injection_point[1], injection_point[0]]):
            border_index = (border_index + 1) % len(perimeter_border)
            injection_point = perimeter_border[border_index]

        injection_point = tuple(injection_point)

        injection_points.append(injection_point)

        cv2.floodFill(window, None, injection_point, i + 1)

    #iterate again and check color order
    color_order = []

    for p in injection_points:

        color_order.append(window[p[1], p[0]])

    #remove unconnected intersections
    i = 0

    while i < len(color_order) - 1:

        if color_order[i] == color_order[i + 1]:

            color_order.pop(i + 1)

            index_of_disconnected_point = (start_index + i + 1) % len(slopes)
            slopes.pop(index_of_disconnected_point)
            if start_index > index_of_disconnected_point:
                start_index -= 1

            for j in range(i, len(color_order)):

                color_order[j] -= 1

        else:
            i += 1

    color_order = tuple(color_order)

    #lookup table for meaning of color order
    color_order_meaning = {(1, 4, 3, 4): 1,
                           (1, 2, 3, 4): 2,
                           (3, 2, 3, 4): 3,
                           (1, 5, 3, 4, 5): 1,  # fused point on perimeter
                           #(1, 2, 3, 4, 5) : 2, #ambiguous case w\ flip
                           # flipped case fuse point on perimeter
                           (4, 2, 3, 4, 5): 4,
                           #(1, 2, 3, 4, 5) : 3 #ambiguous case w\ flip
                           (1, 6, 3, 6, 5, 6): 1,
                           (1, 6, 3, 4, 5, 6): 1,
                           (1, 6, 5, 4, 5, 6): 1,
                           (1, 2, 3, 6, 5, 6): 2,
                           #(1, 2, 3, 4, 5, 6) : 2, #ambiguous case, gets overwritten
                           (1, 2, 5, 4, 5, 6): 2,
                           (3, 2, 3, 6, 5, 6): 3,
                           #(1, 2, 3, 4, 5, 6) : 3, #ambiguous case, might also sometimes mean 2. more often 3 though
                           #symmetric cases for 6 intersections
                           (5, 2, 5, 4, 5, 6): 5,
                           (5, 2, 3, 4, 5, 6): 5,
                           (5, 4, 3, 4, 5, 5): 5,
                           (3, 2, 3, 4, 5, 6): 4,
                           #(1, 2, 3, 4, 5, 6) : 4, #ambiguous case
                           (1, 4, 3, 4, 5, 6): 4
                           }

    #lookup meaning of color order
    if color_order in color_order_meaning:
        point_number = color_order_meaning[color_order]
        return True, slopes[(start_index + point_number) % len(slopes)]

    if len(color_order) == 2:
        return True, slopes[(start_index + 1) % len(slopes)]

    return False, slopes


def _chooseNextPoint(img, r, current_point, last_slope):
    
    x = current_point[0]
    y = current_point[1]

    window = img[y-r: y+r, x-r: x+r]
    next_points = _collectPotentialNextPoints(window, r, r, r)

    if not len(next_points) > 0:
        return None, None

    ret, col_inj_result = _colorInjection(
        np.array([0, 0]) - last_slope, next_points, window)

    if ret:
        last_slope = col_inj_result
        # DEBUG_VISUAL(next_points, img, r, current_point, last_slope)
        current_point += last_slope
    else:
        # DEBUG_points = next_points
        last_slope = _closestSlope(last_slope, col_inj_result)
        # DEBUG_VISUAL(DEBUG_points, img, r, current_point, last_slope)
        current_point += last_slope

    return current_point, last_slope


class _Attempt:
    r = 0

    def __init__(self, r):
        self.r = r


def _traceConnection(img, starting_shape, r):

    start_time = time.time()

    attempts = [_Attempt(r)]  # Attempt(r+2), Attempt(r), Attempt(r-2)]

    starting_point = np.array([starting_shape[1][0], starting_shape[1][1]], dtype=np.int32)#np.array([x_start, y_start], dtype=np.int32)
    current_point = starting_point
    print("START:", starting_point, starting_shape)
    last_slope = np.array([1, 0], dtype=np.int32)

    while(not _isCloseToAnEndpoint(("", tuple(current_point), None)) or _isPointsClose(current_point, starting_point)):
        print(not _isCloseToAnEndpoint(("", tuple(current_point), None)), _isPointsClose(current_point, starting_point))
        if(time.time() - start_time > _TIMEOUT):
            print('TIMEOUT')
            return None

        if current_point[0] - r < 0 or current_point[0] + r > img.shape[1] \
            or current_point[1] - r < 0 or current_point[1] + r > img.shape[0]:
        
            return None

        current_point_votes = []
        last_slope_votes = []

        for att in attempts:
            #look at pixels in circle or square of certain radius around current point
            c, l = _chooseNextPoint(
                img, att.r, current_point, last_slope)

            if c is None:
                return None

            current_point_votes.append(c)
            last_slope_votes.append(l)

        current_point, last_slope = np.array(utils.countVotes(
            current_point_votes)), np.array(utils.countVotes(last_slope_votes))

    return _endpointCloseTo(current_point)

############################################################################

def setWhiteout(wo):
    
    global whiteout
    whiteout = wo

def isConnected(img, r, shape):

    global whiteout

    img = _preprocess(img, whiteout, r)
    intersections = _collectPotentialNextPoints(img, r, shape[1][0], shape[1][1])
    
    return len(intersections) > 0


def traceConnections(img, shapesAndPoints, white_out, r, endpoint_threshold):
    
    global whiteout, _endpoints, _endpoint_threshold

    whiteout = white_out
    _endpoints = shapesAndPoints#[e[1] for e in shapesAndPoints]
    # _endpoints = [point for point in (connection_points[2] for connection_points in shapesAndPoints)]
    print("ENDPOINTS: ", _endpoints)
    
    _endpoint_threshold = endpoint_threshold

    img = _preprocess(img, whiteout, r)

    nodes = shapesAndPoints

    edges = set()

    for p in _endpoints:
        result = _traceConnection(img, p, r)
        if result is not None:
            result_edge = (tuple(result), p)
            if not result_edge in edges:
                edges.add( result_edge )
    
    return (nodes, edges)
