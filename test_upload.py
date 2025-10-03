
import requests
import os

def test_api_upload():
    """Test the API endpoint with a sample image"""
    
    # Create a test image if it doesn't exist
    test_image_path = "uploads/test_img.jpg"
    
    if not os.path.exists(test_image_path):
        print("Please place a test image at 'uploads/test_img.jpg' to run this test")
        return
    
    # Test API endpoint
    url = "http://localhost:5000/api/predict"
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("API Test Successful!")
            print(f"Predicted Disease: {result['predicted_disease']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Disease Info: {result['disease_info']['name']}")
        else:
            print(f"API Test Failed: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Error testing API: {e}")

if __name__ == "__main__":
    test_api_upload()
