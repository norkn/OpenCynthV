from flask import Flask, render_template, Response, request, jsonify, redirect
import cv2
import numpy as np

import main

app = Flask(__name__)


camera = cv2.VideoCapture(1)
frame = None

def gen_frames():  
    global frame

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

@app.route('/test', methods=['GET', 'POST'])
def testfn():
    # GET request
    if request.method == 'GET':
        message = {'greeting':'Hello from Flask!'}
        return jsonify(message)  # serialize and use JSON headers
    # POST request
    if request.method == 'POST':
        print(request.get_json())  # parse as JSON
        return 'Sucesss', 200

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hold')
def hold():
    global frame
    main.hold(frame)
    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True)