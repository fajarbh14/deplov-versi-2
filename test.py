import requests

res = requests.post('https://peakyblinder-lbgplfmbra-as.a.run.app', files={'file': open('./data/test3.png', 'rb')})

print(res.json())