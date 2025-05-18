import pytest
import sys
import os
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app

@pytest.fixture
def client():
    """Create a test client for the app."""
    return TestClient(app)

def test_read_root(client):
    """Test that the root endpoint returns the expected response."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Sentiment API. POST /sentiment/predict"}

def test_app_title():
    """Test that the app has the correct title."""
    assert app.title == "Sentiment API"

def test_router_included():
    """Test that the sentiment router is included in the app."""
    router_paths = [route.path for route in app.routes]
    # Check if any sentiment-related routes exist
    sentiment_routes = [path for path in router_paths if "sentiment" in path]
    assert len(sentiment_routes) > 0

def test_app_initialization():
    """Test that the app is initialized properly."""
    assert isinstance(app, FastAPI)
    assert app.title == "Sentiment API"