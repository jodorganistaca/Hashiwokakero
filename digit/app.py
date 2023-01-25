#!/usr/bin/env python3

import numpy as np
import digitDetector
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

@app.route('/digit', methods=['POST'])
def digit():
    
    data = request.get_json()
 
    img = Image.open(BytesIO(base64.b64decode(data['image'])))
    print("matrix " ,type(img))
    matrix = np.array(data['matrix'])
    resp = {}
    resp["width"] = matrix.shape[0]
    resp["height"] = matrix.shape[1]
    resp["islands"] = []
    resp = digitDetector.decoupe_img(img, matrix, resp)
    print("resp", resp)
    
    return jsonify(resp)


if __name__ == '__main__':
    app.run(host="0.0.0.0")
