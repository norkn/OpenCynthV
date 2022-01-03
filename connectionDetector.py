import math
import numpy as np
import cv2

def DEBUG_VISUAL(slopes, img, r, current_point, last_slope):
    ###########begin visualization for debugging###############
        #print('step ', len(slopes))
        new_img = img.copy()#cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        cv2.rectangle(new_img, tuple(current_point - (r, r)), tuple(current_point + (r, r)), (0, 255, 0))            
                 
        cv2.circle(new_img, tuple(current_point), 2, (0,0,255), thickness = -1)
        
        for s in slopes:
            cv2.circle(new_img, tuple(current_point + s), 2, (0,0,255), thickness = -1)
            
        cv2.circle(new_img, tuple(current_point + last_slope), 3, (255,100,90), thickness = -1)
        
        cv2.imshow('Connection Detection', new_img)
        if cv2.waitKey(50) != -1:
            if cv2.waitKey(200) != -1: return 'break'
        ###########end   visualization for debugging###############

  

  
def isEndpoint(p):
    return False



def isCloseToBlack(c):
    
    threshold = 15
    color_magnitude = c #math.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2)
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



def collectPotentialNextPoints(img, r, x, y):
    perimeter_left   = img[ y+r : y-r : -1,   x-r            ]
    perimeter_top    = img[ y-r           ,   x-r : x+r      ]
    perimeter_right  = img[ y-r : y+r     ,   x+r            ]
    perimeter_bottom = img[ y+r           ,   x+r : x-r : -1 ]
        
    #iterate around perimeter and register intersections
    intersections_left   = collectIntersections(perimeter_left)
    intersections_top    = collectIntersections(perimeter_top)
    intersections_right  = collectIntersections(perimeter_right)
    intersections_bottom = collectIntersections(perimeter_bottom)
        
    #choose most plausible intersection, by slope continuation or color injection   
    slopes = []
        
    for dy in intersections_left   : slopes.append( np.array([  -r, -dy ], dtype=np.int32) )
    for dx in intersections_top    : slopes.append( np.array([  dx,  -r ], dtype=np.int32) )
    for dy in intersections_right  : slopes.append( np.array([   r,  dy ], dtype=np.int32) )
    for dx in intersections_bottom : slopes.append( np.array([ -dx,   r ], dtype=np.int32) )
    
    return slopes



def processImageForSlopeFollowing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations = 2)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    img = cv2.erode(img, kernel, iterations = 2)
    img = cv2.dilate(img, kernel, iterations = 2)
    #img = cv2.medianBlur(img, 9)
    img = cv2.dilate(img, kernel, iterations = 1)
    
    return img



def processImageForColorInjection(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations = 2)
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    img = cv2.erode(img, kernel, iterations = 2)
    img = cv2.dilate(img, kernel, iterations = 2)
    img = cv2.medianBlur(img, 9)
    
    return img



def colorInjection(last_slope, slopes, window):
    
    window = window.copy()
    
    #find last_slope in slopes
    min_distance = 100000
    start_index = -1
    
    for i in range(len(slopes)):
    
        d = np.linalg.norm(slopes[i] - last_slope)
        
        if d < min_distance:
            min_distance = d
            start_index = i
            
    #iterate from there through slopes
    injection_points = []
    
    for i in range(len(slopes)):
        
        index = (start_index + i) % len(slopes)
        
        #inject color
        p = slopes[index]
        r = int(window.shape[0] / 2)
        
        step = (-int(p[1] / r), int(p[0] / r))#(int(p[0] / r), int(p[1] / r))
        
        if step == (-1, -1):
            step = (1, 0)
        elif step == (1, -1):
            step = (0, 1)
        elif step == (1, 1):
            step = (-1, 0)  
        elif step == (-1, 1):
            step = (0, -1)
        
        injection_point = p + (r, r) + step
        
        if(injection_point[0] == 2 * r):
            injection_point[0] = 2 * r - 1
        if(injection_point[1] == 2 * r):
            injection_point[1] = 2 * r - 1
            
        injection_point = tuple(injection_point)
            
        injection_points.append(injection_point)
        
        cv2.floodFill(window, None, injection_point, i + 1)
        
    #iterate again and check color order
    color_order = []
    
    for p in injection_points:
        
        color_order.append(window[p[0], p[1]])
        
    #lookup meaning of color order
    if color_order == [1, 4, 3, 4]:
        return slopes[start_index + 1]
    elif color_order == [1, 2, 3, 4]:
        return slopes[start_index + 2]
    elif color_order == [3, 2, 3, 4]:
        return slopes[start_index + 3]
    
    return -1


   
def traceConnection(img, x_start, y_start, x_end, y_end, r):
    
    current_point = np.array([x_start, y_start], dtype=np.int32)
    #pad image
    last_slope = np.array([1, 0], dtype=np.int32)
    
    while(not isEndpoint(current_point)):
    
        #look at pixels in circle or square of certain radius around current point 
        x = current_point[0]
        y = current_point[1]
        
        slopes = collectPotentialNextPoints(img, r, x, y)
        
        
        
        last_slope = closestSlope(last_slope, slopes)
        
        if DEBUG_VISUAL(slopes, img, r, current_point, last_slope) == 'break': break
        
        current_point += last_slope   #last_slope is still current slope at this point but already called last_slope for next step
          
    return current_point

def mouseCallback(event, x, y, flags, param):
    global video, isRunning
    
    ret, img = video.read()
    #img = processImageForSlopeFollowing(img)
    img = processImageForColorInjection(img)
    if event == cv2.EVENT_LBUTTONDOWN:
        traceConnection(img, x, y, 135, 50, 6)
        isRunning = True
    
    

video = cv2.VideoCapture(0)
ret, img = video.read()

fps = video.get(cv2.CAP_PROP_FPS)

cv2.namedWindow('Connection Detection')
cv2.setMouseCallback('Connection Detection', mouseCallback)

isRunning = False

while video.isOpened():
    
    if not isRunning:
        ret, frame = video.read()
    
        cv2.imshow('Connection Detection', frame)
    
    if cv2.waitKey(int(1000 / fps) + 1) != -1:
        break

cv2.destroyAllWindows()
