#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 08:01:32 2022

@author: mocquais
"""

import numpy as np
import isleDetector
from flask import Flask, request, jsonify
import cv2 as cv2
import base64


app = Flask(__name__)

@app.route('/island', methods=["POST"])
def test():
    data = request.get_json(silent=True)
    decoded_image = base64.b64decode(data['image'])
    
    img = cv2.imdecode(np.frombuffer(decoded_image, np.uint8), cv2.IMREAD_GRAYSCALE)

    matrix = np.array(isleDetector.detect_circles(img))
    
    return jsonify({'matrix': matrix.tolist()})


if __name__ == '__main__':
    app.run(host="0.0.0.0")
