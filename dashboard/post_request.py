import requests

with open('example.json', 'rb') as f:
	data = f.read()
#     r = requests.post("http://0.0.0.0:8080/score", files={'example.json': f})

r = requests.post("http://0.0.0.0:8080/score", json=data)