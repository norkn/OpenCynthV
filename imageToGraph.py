import cv2
import numpy as np

import Module_final as sh
import connectionDetector as cd
import utils


def _getWhiteoutByHue(img, hue, threshold):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h, s, v =  cv2.split(img)
    
    h_whiteout = cv2.inRange(h, hue - threshold, hue + threshold)

    lower_hsv = np.array([hue - threshold,0,0])
    upper_hsv = np.array([hue + threshold,255,255])
    
    h_whiteout = cv2.inRange(img, lower_hsv, upper_hsv)

    return h_whiteout


def _applyWhiteout(whiteout, frame):

    no_connections_mask = cv2.bitwise_and(whiteout, sh._preprocess(frame))
    no_connections_mask_inverted = cv2.bitwise_not(no_connections_mask)

    kernel = np.ones((3, 3), np.uint8)
    no_connections_mask_inverted = cv2.erode(no_connections_mask_inverted, kernel, iterations=3)

    no_connections_mask_inverted = cv2.merge((no_connections_mask_inverted, no_connections_mask_inverted, no_connections_mask_inverted))
    frame_modules_only = cv2.bitwise_or(frame, no_connections_mask_inverted)
    
    return frame_modules_only


def _updateGraph(graph, removed_nodes, added_edges):

    for node in removed_nodes:
        
        if node in graph and graph[node] in graph:
            graph.pop(graph[node])
        if node in graph:
            graph.pop(node)

    for edge in added_edges:

        graph[edge[0]] = edge[1]
        graph[edge[1]] = edge[0]

    return graph


def registerModules(frame, hue, hue_treshold, min_area, graph):

    graph.clear()

    modules_whiteout = _getWhiteoutByHue(frame, hue, hue_treshold)
    frame_modules_only = _applyWhiteout(modules_whiteout, frame)
    #detect modules
    shapes, shape_contours = sh.findShapes(frame_modules_only, frame_modules_only.shape[1], frame_modules_only.shape[0], min_area)
    last_state_was_connected = [False] * len(shapes)

    return shapes, shape_contours, last_state_was_connected


def updateConnections(frame, hue, hue_threshold, shapes, last_state_was_connected, r, endpoint_vicinity, graph):

    #go over all shapes and see if connection needs to be traced
    shapes_to_trace = []
    new_connected_states = []
    disconnected_nodes = []

    modules_whiteout = _getWhiteoutByHue(frame, hue, hue_threshold)
    cd.setWhiteout(modules_whiteout)

    for i in range(len(shapes)):

            new_connected_states.append(cd.isConnected(frame, r, shapes[i]))

            if new_connected_states[i] and not last_state_was_connected[i]:
                shapes_to_trace.append(shapes[i])
            elif not new_connected_states[i] and last_state_was_connected[i]:
                disconnected_nodes.append(shapes[i])

    last_state_was_connected = new_connected_states
    shapes_to_trace = shapes ###########################################

    nodes, _new_edges = cd.traceConnections(frame, shapes_to_trace, modules_whiteout, r, endpoint_vicinity)
    graph = _updateGraph(graph, disconnected_nodes, _new_edges)

    print("GRAPH:",graph)

    return graph