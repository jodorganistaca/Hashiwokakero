from flask import Flask, render_template, request, redirect, url_for
import json
import requests
import cv2 as cv2
import numpy as np
import base64
import imutils

global IP_SERVER
IP_SERVER = 'http://192.168.37.186:'

global ISLAND_PORT
ISLAND_PORT = 50002

global ISLAND_ENDPOINT
ISLAND_ENDPOINT = '/island'

global DIGIT_PORT
DIGIT_PORT = 50030

global DIGIT_ENDPOINT
DIGIT_ENDPOINT = '/digit'

global SOLVER_PORT
SOLVER_PORT = 50040

global SOLVER_ENDPOINT
SOLVER_ENDPOINT = '/solve'

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']

        if uploaded_file.filename != '':
            uploaded_file.save(uploaded_file.filename)
            # Prepare image
            img = cv2.imread(uploaded_file.filename, cv2.IMREAD_GRAYSCALE)
            resized_img = imutils.resize(img,width=1000)

            _, buffer = cv2.imencode('.jpg', resized_img)
            jpg_as_text = base64.b64encode(buffer).decode()
            with open('out.txt', 'w') as f:
                print('{' + "image" + ':' + jpg_as_text + '}', file=f)
            data = {"image": jpg_as_text}
            header = {"Content-Type": "application/json"}
            # send image to island recognition
            print('***POST ON : ' + IP_SERVER +
                  str(ISLAND_PORT) + ISLAND_ENDPOINT + '***\n')
            island_res = requests.post(
                 IP_SERVER + str(ISLAND_PORT) + ISLAND_ENDPOINT, data=json.dumps(data), headers=header)
            if island_res.status_code >= 400:
                raise Exception("ERROR WITH island")
            island_matrix = island_res.json()
            island_matrix["image"]=jpg_as_text
            print(island_matrix)

            # Prepare and send data for digit recognition
            print('*** POST ON : ' + IP_SERVER +
                  str(DIGIT_PORT) + DIGIT_ENDPOINT + '***\n')
            digit_res = requests.post(
                IP_SERVER + str(DIGIT_PORT) + DIGIT_ENDPOINT, json=island_matrix)
            if digit_res.status_code > 400:
                raise Exception("ERROR WITHT DIGIT RECOGNITION\n")
            digit_json = digit_res.json()
            print(digit_json)
            # # Prepare and send data for solver
            print('*** POST ON : ' + IP_SERVER +
                  str(SOLVER_PORT) + SOLVER_ENDPOINT + '***\n')
            solver_res = requests.post(
                IP_SERVER + str(SOLVER_PORT) + SOLVER_ENDPOINT, json=digit_json)
            if solver_res.status_code > 400:
                raise Exception("ERROR WITHT DIGIT RECOGNITION\n")
            print(solver_res.text)

        # return redirect(url_for('index'))
        return render_template('index.html',result=solver_res.text)
    return render_template('index.html')
