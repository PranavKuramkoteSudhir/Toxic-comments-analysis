import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    rv = client.get('/')
    assert rv.status_code == 200

def test_predict_api(client):
    response = client.post('/api/predict', 
                          json={'comment_text': 'test comment'})
    assert response.status_code == 200