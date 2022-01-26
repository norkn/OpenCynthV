import cv2


class UI:

    _RESOLUTION = 1000

    paramNames = None
    paramRanges = None
    paramRefs = None


    def callbackForParam(self, i):
        
        def callback(v):
            min = self.paramRanges[i][0]
            max =  self.paramRanges[i][1]
            self.paramRefs[i] = min + (v / self._RESOLUTION) * (max - min)

        return callback


    def __init__(self, paramNames, paramRanges, paramRefs):

        self.paramNames = paramNames
        self.paramRanges = paramRanges
        self.paramRefs = paramRefs

        cv2.namedWindow('UI')

        for i in range(len(self.paramRefs)):
            
            min = self.paramRanges[i][0]
            max =  self.paramRanges[i][1]
            default = int((paramRefs[i] - min)/ (max - min) * self._RESOLUTION)

            cv2.createTrackbar(self.paramNames[i], 'UI', default, self._RESOLUTION, self.callbackForParam(i))

