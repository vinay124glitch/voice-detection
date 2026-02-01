import requests
import base64
import json
import requests
import base64
import json
import io

# Minimal valid MP3 frame (MPEG 1 Layer 3, 44.1kHz, 32kbps) - 104 bytes
# This avoids dependencies on local ffmpeg/wav encoders for the test.
DUMMY_AUDIO_B64 = "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4LjI5LjEwMAAAAAAAAAAAAAAA//OEAAAAAAA0gAAAAAAAAAAAAAAAAAAAAA=="

API_URL = "http://127.0.0.1:8000/api/voice-detection"
API_KEY = "sk_voice_9f83kls92k"

def test_voice_detection():
    print(f"Testing API at {API_URL}...")
    
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": DUMMY_AUDIO_B64
    }
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        try:
            data = response.json()
            print("Response Body:")
            print(json.dumps(data, indent=2))
        except:
            print("Raw Response:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Is it running? (uvicorn main:app --reload)")

if __name__ == "__main__":
    # We need requests to run this script
    try:
        import requests
    except ImportError:
        print("Installing requests library for the test script...")
        import subprocess
        subprocess.check_call(["pip", "install", "requests"])
        import requests

    test_voice_detection()
