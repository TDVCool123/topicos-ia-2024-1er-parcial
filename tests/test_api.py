from fastapi.testclient import TestClient
from src.main import app


client = TestClient(app)



def test_detect_people():
    with open("gun1.jpg", "rb") as img_file:
        files = {"file": ("test_image.jpg", img_file, "image/jpeg")}
        response = client.post("/detect_people", files=files)

    assert response.status_code == 200
    assert "labels" in response.json()
    assert "boxes" in response.json()

def test_annotate_people():
    with open("gun1.jpg", "rb") as img_file:
        files = {"file": ("test_image.jpg", img_file, "image/jpeg")}
        response = client.post("/annotate_people", files=files)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

def test_detect():
    with open("gun1.jpg", "rb") as img_file:
        files = {"file": ("test_image.jpg", img_file, "image/jpeg")}
        response = client.post("/detect", files=files)

    assert response.status_code == 200
    assert "detection" in response.json()
    assert "segmentation" in response.json()

def test_annotate():
    with open("gun1.jpg", "rb") as img_file:
        files = {"file": ("test_image.jpg", img_file, "image/jpeg")}
        response = client.post("/annotate", files=files)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"

def test_guns():
    with open("gun1.jpg", "rb") as img_file:
        files = {"file": ("test_image.jpg", img_file, "image/jpeg")}
        response = client.post("/guns", files=files)

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    for gun in response.json():
        assert "type" in gun
        assert "location" in gun

def test_people():
    with open("gun1.jpg", "rb") as img_file:
        files = {"file": ("test_image.jpg", img_file, "image/jpeg")}
        response = client.post("/people", files=files)

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    for person in response.json():
        assert "category" in person
        assert "location" in person
        assert "area" in person
