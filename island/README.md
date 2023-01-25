Pour lancer l'application flask, il faut crée la variable d'environnement FLASK_APP avec le nom de notre application :

export FLASK_APP=isleDetectorFlask.py

Puis, vous pourrez la lancer en utilisant flask run

l'application en elle même ecoute sur localhost:5000 donc il faudra rediriger le flux entrant vers cette adresse.

Les librairies a installé pour le bon fonctionnement du programme sont les suivantes :
``` Python
import numpy as np
import cv2 as cv2
import imutils as imutils
from flask import Flask, request, jsonify
import base64

La requête envoyer à l'app doit être de la forme : 

import json
import requests
import cv2 as cv2
import numpy as np
import base64

img = cv2.imread('generator/Dataset/Test/HandDrawn/1.jpg',cv2.IMREAD_GRAYSCALE) #Cette image peut aussi être récupérée autrement puis transformée en image cv2

_, buffer = cv2.imencode('.jpg', img)
jpg_as_text = base64.b64encode(buffer).decode()

data = json.dumps({'image': jpg_as_text})
headers = {'Content-type': 'application/json'}

response = requests.post('http://127.0.0.1:5000/', data=data, headers=headers)


La réponse de l'app peut être déchiffrer de cette manière :

result = response.json()

decoded_image = base64.b64decode(result['image'])

img = cv2.imdecode(np.frombuffer(decoded_image, np.uint8), cv2.IMREAD_GRAYSCALE)

cv2.imshow('detected circles',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(result['matrix'])
```
