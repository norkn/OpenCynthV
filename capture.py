import json
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import imageToGraph as g
import main

app = Flask(__name__)


camera = cv2.VideoCapture(0)

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame

        frame = main.update(frame, main.params)

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('OpenCynthV.html')

@app.route('/getGraph', methods=['GET'])
def getGraph():
    print(json.dumps(main.getGraph))
    return json.dumps(main.getGraph)
    
    
    
@app.route('/holdContours', methods=['POST'])
def holdContours():
    if request.method == 'POST':
        #main.holdContours()
        return "OK"
    else:
        return "NO OK"
    

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)