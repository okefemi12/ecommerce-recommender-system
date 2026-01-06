import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.main import app

# Initialize the Test Client
client = TestClient(app)

def test_api_health_check():
    """
    Test 1: Verify the API is awake and reachable.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is running!"}

@patch("src.model.ModelService.predict")
@patch("src.model.ModelService.interpreter") 
def test_recommendation_endpoint(mock_interpreter, mock_predict):
    
    # --- SETUP MOCK ---
    # 1. Pretend the interpreter is loaded (not None)
    mock_interpreter.return_value = True 
    
    # 2. Pretend the model predicts Product ID #5 with some confidence scores
    # Returns: (Best Action, List of Q-Values)
    mock_predict.return_value = (5, [0.1, 0.05, 0.8, 0.05]) 

    # --- EXECUTE ---
    payload = {"history": [101, 202, 303]}
    response = client.post("/recommend", json=payload)

    # --- ASSERT (Check Results) ---
    assert response.status_code == 200
    data = response.json()
    
    # Check if we got the expected structure
    assert "recommended_id" in data
    assert "all_q_values" in data
    
    # Check if our Mock logic worked (Did it return product 5?)
    assert data["recommended_id"] == 5

def test_invalid_payload():
    """
    Test 3: Verify the API rejects bad data (e.g., missing history).
    """
    # Send empty JSON
    response = client.post("/recommend", json={})
    
    # Should fail with 422 Unprocessable Entity (FastAPI standard error)
    assert response.status_code == 422