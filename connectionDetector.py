import math
import numpy as np
import cv2

def DEBUG_VISUAL(slopes, img, r, current_point, last_slope):
    ###########begin visualization for debugging###############
    global input_img
    new_img = input_img.copy()#cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)#img.copy()#cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    cv2.rectangle(new_img, tuple(current_point - (r, r)), tuple(current_point + (r, r)), (0, 255, 0))            
             
    cv2.circle(new_img, tuple(current_point), 2, (0,0,255), thickness = -1)
    
    for s in slopes:
        cv2.circle(new_img, tuple(current_point + s), 2, (0,0,255), thickness = -1)
        
    cv2.circle(new_img, tuple(current_point + last_slope), 3, (255,100,90), thickness = -1)
    
    cv2.imshow('Connection Detection', new_img)
    if cv2.waitKey(50) != -1:
        if cv2.waitKey(20) != -1:
            cv2.destroyAllWindows()
            quit()
    ###########end   visualization for debugging###############

  

inf = 1000000
  
def isEndpoint(p):
    return False



def isCloseToBlack(c):
    
    threshold = 0.5
    color_magnitude = c #math.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2)
    return color_magnitude < threshold


  
def getPerimeterBorder(r):
    perimeter_left   = [np.array([0        , 2*r-1 - i]) for i in range(2*r)]
    perimeter_top    = [np.array([i        , 0        ]) for i in range(2*r)]
    perimeter_right  = [np.array([2*r-1    , i        ]) for i in range(2*r)]    
    perimeter_bottom = [np.array([2*r-1 - i, 2*r-1    ]) for i in range(2*r)]
    
    return perimeter_left + perimeter_top + perimeter_right + perimeter_bottom



def collectPotentialNextPoints(img, r, x, y):
    
    perimeter_border = getPerimeterBorder(r)
    
    intersections = []
    
    isOnIntersection = False
    avg_index = 0
    i = 0
    count = 0
        
    for p in perimeter_border:
            
        i += 1
            
        if isCloseToBlack(img[y - r + p[1], x - r + p[0]]):
               
            isOnIntersection = True
            avg_index += i
            count += 1
                
        else:
                
            if isOnIntersection:
                    
                isOnIntersection = False
                avg_index = int(avg_index / count)
                intersections.append(perimeter_border[avg_index] - np.array([r, r]))
                avg_index = 0
                count = 0
                
    return intersections
    
    

def indexOfClosestElementInList(e, l):
    
    min_distance = inf
    index = -1
    
    for i in range(len(l)):
    
        d = np.linalg.norm(l[i] - e)
        
        if d < min_distance:
            min_distance = d
            index = i
            
    return index



def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((3, 3), np.uint8)

    img = cv2.GaussianBlur(img,(3,3),8)
    img = cv2.erode(img, kernel, iterations = 1)   
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,4)    
    img = cv2.erode(img, kernel, iterations = 2)
    img = cv2.medianBlur(img, 5)
    img = cv2.dilate(img, kernel, iterations = 2)
    
    return img



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
        
        

def colorInjection(last_slope, slopes, window):
    
    window = window.copy()
    
    start_index = indexOfClosestElementInList(closestSlope(last_slope, slopes), slopes) #indexOfClosestElementInList(last_slope, slopes)
    r = int(window.shape[0] / 2)
    perimeter_border = getPerimeterBorder(r)
    
    injection_points = []
    
    #inject color 
    for i in range(len(slopes)):
        
        index = (start_index + i) % len(slopes)
        
        injection_point = slopes[index] + (r, r)
        border_index = indexOfClosestElementInList(injection_point, perimeter_border)
        
        while isCloseToBlack(window[injection_point[1], injection_point[0]]):
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
            if start_index > index_of_disconnected_point: start_index -= 1
            print(start_index)
            for j in range(i, len(color_order)):
                               
                color_order[j] -= 1
                
        else:
            i += 1
            
    
    color_order = tuple(color_order)
        
    #lookup table for meaning of color order
    color_order_meaning = {(1, 4, 3, 4) : 1, 
                           (1, 2, 3, 4) : 2,
                           (3, 2, 3, 4) : 3,
                           (1, 5, 3, 4, 5) : 1, #fused point on perimeter
                           #(1, 2, 3, 4, 5) : 2, #ambiguous case w\ flip
                           (4, 2, 3, 4, 5) : 4, #flipped case fuse point on perimeter
                           #(1, 2, 3, 4, 5) : 3 #ambiguous case w\ flip
                           (1, 6, 3, 6, 5, 6) : 1,
                           (1, 6, 3, 4, 5, 6) : 1,
                           (1, 6, 5, 4, 5, 6) : 1,
                           (1, 2, 3, 6, 5, 6) : 2,
                           #(1, 2, 3, 4, 5, 6) : 2, #ambiguous case, gets overwritten
                           (1, 2, 5, 4, 5, 6) : 2,
                           (3, 2, 3, 6, 5, 6) : 3,
                           #(1, 2, 3, 4, 5, 6) : 3, #ambiguous case, might also sometimes mean 2. more often 3 though
                           #symmetric cases for 6 intersections
                           (5, 2, 5, 4, 5, 6) : 5,
                           (5, 2, 3, 4, 5, 6) : 5,
                           (5, 4, 3, 4, 5, 5) : 5,
                           (3, 2, 3, 4, 5, 6) : 4,
                           #(1, 2, 3, 4, 5, 6) : 4, #ambiguous case
                           (1, 4, 3, 4, 5, 6) : 4               
                           }
            
    
    #lookup meaning of color order
    if color_order in color_order_meaning:
        point_number = color_order_meaning[color_order]
        return True, slopes[(start_index + point_number)%len(slopes)]
    
    if len(color_order) == 2:
        return True, slopes[(start_index + 1)%len(slopes)]
    
    return False, slopes



def chooseNextPoint(img, r, current_point, last_slope):
    
    x = current_point[0]
    y = current_point[1]
    
    window = img[ y-r : y+r, x-r : x+r ]
    next_points = collectPotentialNextPoints(window, r, r, r)
    
    ret, col_inj_result = colorInjection(np.array([0, 0]) - last_slope, next_points, window)
    
    if ret:
        last_slope = col_inj_result
        DEBUG_VISUAL(next_points, img, r, current_point, last_slope)
        current_point += last_slope
    else:
        DEBUG_points = next_points
        next_points = collectPotentialNextPoints(img, r, x, y)
        last_slope = closestSlope(last_slope, col_inj_result)#next_points)
        DEBUG_VISUAL(DEBUG_points, img, r, current_point, last_slope)
        current_point += last_slope
    
    return current_point, last_slope



def countVotes(l):
    
    dictionary = {}
    
    for element in l:
        
        element = tuple(element)
        
        if element in dictionary: 
            dictionary[element] += 1
        else:      
            dictionary[element] = 1
      
    result = None
    max_votes = -1
    
    for element in dictionary.keys():
        
        if dictionary[element] > max_votes:
            result = element
            max_votes = dictionary[element]
            
    return result



class Attempt:
    r = 0
    
    def __init__(self, _r):
            self.r = _r
         
 
           
def traceConnection(img, x_start, y_start, x_end, y_end, r):
    
    attempts = [Attempt(r)]#Attempt(r+2), Attempt(r), Attempt(r-2)]
    
    img_color_injection = preprocess(img)
    
    current_point = np.array([x_start, y_start], dtype=np.int32)
    #pad image
    last_slope = np.array([1, 0], dtype=np.int32)
    
    while(not isEndpoint(current_point)):
    
        current_point_votes = []
        last_slope_votes = []
        
        for att in attempts:
            #look at pixels in circle or square of certain radius around current point 
            c, l = chooseNextPoint(img_color_injection, att.r, current_point, last_slope)
            current_point_votes.append(c)
            last_slope_votes.append(l)
        
        current_point, last_slope = np.array(countVotes(current_point_votes)), np.array(countVotes(last_slope_votes))        
        
    return current_point



#########################################


def mouseCallback(event, x, y, flags, param):

    global input_img
    if event == cv2.EVENT_LBUTTONDOWN:
        traceConnection(input_img, x, y, 135, 50, 26)
        isRunning = True
    
    
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
