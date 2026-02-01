
# REST API for AI-Generated Voice Detection
# Uses FastAPI, Librosa, and basic heuristics for demonstration.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: MP3 processing requires `ffmpeg` on the system. The Dockerfile handles this automatically.*

2. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

## API Usage

### Endpoint: `POST /api/voice-detection`

**Headers:**
- `x-api-key`: `sk_voice_9f83kls92k`

**Body:**
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<BASE64_STRING>"
}
```

**Response:**
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.95,
  "explanation": "High spirality and low jitter variance detected."
}
```

## Deployment

### Option 1: Docker (Recommended)
Since audio processing requires system libraries like `ffmpeg`, using Docker is the most reliable way to deploy.

1. **Build the image:**
   ```bash
   docker build -t voice-detection-api .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 voice-detection-api
   ```

### Option 2: Render
1. Push this repository to GitHub/GitLab.
2. Link your repository to Render.
3. **Select "Docker"** as the environment (Render will automatically detect the `Dockerfile`).
4. or use the `render.yaml` Blueprint for 1-click configuration.
