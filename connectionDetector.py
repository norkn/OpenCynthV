import math
import numpy as np
import cv2
    
def isEndpoint(p):
    return False


def isCloseToBlack(c):
    
    threshold = 15
    color_magnitude = math.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2)
    return color_magnitude < threshold
    

def closestSlope(slope, slopes):
    
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
        
        
    
def collectIntersections(perimeter_border):
   
    intersections = []
    
    isOnIntersection = False
    avg_coord = 0
    i = 0
    count = 0
        
    for p in perimeter_border:
            
        i += 1
            
        if(isCloseToBlack(p)):
               
            isOnIntersection = True
            avg_coord += i
            count += 1
                
        else:
                
            if isOnIntersection:
                    
                isOnIntersection = False
                avg_coord /= count
                avg_coord -= len(perimeter_border) / 2
                intersections.append(avg_coord)
                avg_coord = 0
                count = 0
                
    if isOnIntersection:    #if the side ends on an intersection, we still need to register it
        avg_coord /= count
        avg_coord -= len(perimeter_border) / 2
        intersections.append(avg_coord)
                
    return intersections



def traceConnection(img, x_start, y_start, x_end, y_end, r):
    
    current_point = np.array([x_start, y_start], dtype=np.int32)
    #pad image
    last_slope = np.array([1, 0], dtype=np.int32)
    
    while(not isEndpoint(current_point)):
    
        #look at pixels in circle or square of certain radius around current point 
        x = current_point[0]
        y = current_point[1]
        
        perimeter_left   = img[ y - r : y + r,   x - r         ]
        perimeter_top    = img[ y - r        ,   x - r : x + r ]
        perimeter_right  = img[ y - r : y + r,   x + r         ]
        perimeter_bottom = img[ y + r        ,   x - r : x + r ]
        
        #iterate around perimeter and register intersections
        intersections_left   = collectIntersections(perimeter_left)
        intersections_top    = collectIntersections(perimeter_top)
        intersections_right  = collectIntersections(perimeter_right)
        intersections_bottom = collectIntersections(perimeter_bottom)
        
        #choose most plausible intersection, by slope continuation or color injection   
        slopes = []
        
        for dy in intersections_left   : slopes.append( np.array([ -r, dy ], dtype=np.int32) )
        for dx in intersections_top    : slopes.append( np.array([ dx, -r ], dtype=np.int32) )
        for dy in intersections_right  : slopes.append( np.array([  r, dy ], dtype=np.int32) )
        for dx in intersections_bottom : slopes.append( np.array([ dx,  r ], dtype=np.int32) )
        
        last_slope = closestSlope(last_slope, slopes)
        
        ###########begin visualization for debugging###############
        print('step')
        new_img = img.copy()
        
        cv2.rectangle(new_img, tuple(current_point - (r, r)), tuple(current_point + (r, r)), (0, 255, 0))            
                 
        cv2.circle(new_img, tuple(current_point), 2, (0,0,255), thickness = -1)
        
        for s in slopes:
            cv2.circle(new_img, tuple(current_point + s), 2, (0,0,255), thickness = -1)
            
        cv2.circle(new_img, tuple(current_point + last_slope), 2, (255,100,90), thickness = -1)
        
        cv2.imshow('Connection Detection', new_img)
        if cv2.waitKey(0) != -1:
            if cv2.waitKey(150) != -1: break
        ###########end   visualization for debugging###############
        
        current_point += last_slope   #last_slope is still current slope at this point but already called last_slope for next step
        
    
    return current_point

def mouseCallback(event, x, y, flags, param):
    
    if event == cv2.EVENT_LBUTTONDOWN: traceConnection(erosion, x, y, 135, 50, 6)
    
img = cv2.imread('connections_test.jpg', cv2.IMREAD_COLOR)
cv2.namedWindow('Connection Detection')
cv2.setMouseCallback('Connection Detection', mouseCallback)

kernel = np.ones((3, 3), np.uint8)
erosion = cv2.dilate(img, kernel, iterations = 1)
cv2.imshow('Connection Detection', img)

#connection extraction
#color threshold
#morphological filtering

cv2.destroyAllWindows()
