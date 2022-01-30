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

    out_graph = {}

    if graph is not None:
        for key in graph.keys():
            out_graph[str(key)] = str(graph[key])
    
    return jsonify(json.loads(jsonpickle.encode(out_graph)))

if __name__ == "__main__":
    app.run(debug=True)




# @app.route('/test', methods=['GET', 'POST'])
# def testfn():
#     # GET request
#     if request.method == 'GET':
#         m = Module("osc", (1,2), [(0,0), (20,20)])
#         m2 = Module("osc", (12,2), [(0,0), (20,20)])

#         c  = Connection(m,m2)

#         return jsonify(json.loads(jsonpickle.encode(c)))#jsonify(c)  # serialize and use JSON headers
#     # POST request
#     if request.method == 'POST':
#         print(request.get_json())  # parse as JSON
#         return 'Sucesss', 200