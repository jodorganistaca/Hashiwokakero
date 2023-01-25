# Solver

- Run `python solve_api.py`
- Send POST requests with json files representing the grid that needs to be solved

Example : 

	import requests
	import json
	import numpy as np

	grid = json.load(open(path_to_json_file, 'r'))
	res = requests.post('http://localhost:5000/solve', json=grid)

	if res.ok:
	    print(res.text)

