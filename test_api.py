import requests

data = {"features":[63,1,3,145,233,1,0,150,0,2.3,0,0,1]}  # example values
r = requests.post("http://127.0.0.1:5000/predict", json=data)
print(r.status_code, r.json())
