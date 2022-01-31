from flask import Flask, render_template, Response, request, jsonify, redirect
import jsonpickle, json
import cv2

import main
from modulesAndConnections import *

app = Flask(__name__)


camera = cv2.VideoCapture(0)

frame = None
graph = None


def gen_frames():  
    global frame, graph

    while True:
        success, frame = camera.read()

        if not success:
            break
        else:
            frame = main.drawContoursAndConnections(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('OpenCynthV.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update')
def update():
    global frame, graph
    if frame is not None:
        graph = main.updateGraph(frame, main.params)
    return redirect('/')

@app.route('/hold')
def hold():
    global frame
    main.hold(frame)
    return redirect('/')

@app.route('/get_graph')
def getGraph():
    global graph    
    return jsonify(json.loads(jsonpickle.encode(graph)))

if __name__ == "__main__":
    app.run(debug=True)