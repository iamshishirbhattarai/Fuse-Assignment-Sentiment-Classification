import pytest
import sys
import os
import torch
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import after path fix
from app.sentiment import router, predict_sentiment
from app.schema import SentimentRequest, SentimentResponse
from app.main import app

# Create a test client
@pytest.fixture
def client():
    return TestClient(app)

# Mock the transformer model outputs
@pytest.fixture
def mock_model():
    model_mock = MagicMock()
    output_mock = MagicMock()
    # Create a mock for the logits property
    output_mock.logits = torch.tensor([[0.1, 0.9]])  # positive sentiment
    model_mock.return_value = output_mock
    return model_mock

@pytest.fixture
def mock_negative_model():
    model_mock = MagicMock()
    output_mock = MagicMock()
    # Create a mock with negative sentiment logits
    output_mock.logits = torch.tensor([[0.9, 0.1]])  # negative sentiment
    model_mock.return_value = output_mock
    return model_mock

def test_predict_endpoint_structure(client):
    """Test the structure of the predict endpoint."""
    with patch('app.sentiment.model') as mock_model:
        # Configure the mock
        output = MagicMock()
        output.logits = torch.tensor([[0.1, 0.9]])
        mock_model.return_value = output
        
        # Test the endpoint
        response = client.post("/sentiment/predict", json={"text": "I love this movie"})
        assert response.status_code == 200
        
        # Check response structure
        result = response.json()
        assert "label" in result
        assert "scores" in result
        assert "negative" in result["scores"]
        assert "positive" in result["scores"]

def test_empty_text_validation(client):
    """Test that empty text raises a 400 error."""
    response = client.post("/sentiment/predict", json={"text": ""})
    assert response.status_code == 400
    assert "must not be empty" in response.json()["detail"]

def test_positive_sentiment_prediction(client):
    """Test positive sentiment prediction."""
    with patch('app.sentiment.model') as mock_model:
        # Configure the mock
        output = MagicMock()
        output.logits = torch.tensor([[0.1, 0.9]])
        mock_model.return_value = output
        
        response = client.post("/sentiment/predict", json={"text": "I love this movie"})
        result = response.json()
        assert result["label"] == "positive"
        assert result["scores"]["positive"] > result["scores"]["negative"]

def test_negative_sentiment_prediction(client):
    """Test negative sentiment prediction."""
    with patch('app.sentiment.model') as mock_model:
        # Configure the mock
        output = MagicMock()
        output.logits = torch.tensor([[0.9, 0.1]])
        mock_model.return_value = output
        
        response = client.post("/sentiment/predict", json={"text": "I hate this movie"})
        result = response.json()
        assert result["label"] == "negative"
        assert result["scores"]["negative"] > result["scores"]["positive"]

def test_predict_sentiment_function():
    """Test the predict_sentiment function directly."""
    with patch('app.sentiment.model') as mock_model:
        # Configure the mock
        output = MagicMock()
        output.logits = torch.tensor([[0.1, 0.9]])
        mock_model.return_value = output
        
        request = SentimentRequest(text="This is a great product!")
        result = predict_sentiment(request)
        
        assert isinstance(result, SentimentResponse)
        assert result.label == "positive"
        assert result.scores["positive"] > result.scores["negative"]

def test_model_inference_exception():
    """Test handling of exceptions during model inference."""
    with patch('app.sentiment.model') as mock_model:
        # Make the model raise an exception when called
        mock_model.side_effect = RuntimeError("Model inference error")
        
        with pytest.raises(Exception):
            request = SentimentRequest(text="Test text")
            predict_sentiment(request)