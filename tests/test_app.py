from app.app import app

def test_predict_success():
    client = app.test_client()
    response = client.post('/predict', json={'features': [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    assert 'prediction' in response.get_json()

def test_predict_fail():
    client = app.test_client()
    response = client.post('/predict', json={})
    assert response.status_code == 400
