import requests

BASE_URL = "http://192.168.49.2:30969"

def test_root():
    response = requests.get(BASE_URL + "/")
    assert response.status_code == 200
    assert "MLOps Flask App is Live!" in response.text

def test_prediction():
    data = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = requests.post(BASE_URL + "/predict", json=data)
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
def test_dummy():
    assert 1 + 1 == 2
