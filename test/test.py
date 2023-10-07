import requests

res = requests.post('http://127.0.0.1:5000/', files={'file': open('../data/paangga3.jpg', 'rb')})

print(res.json())