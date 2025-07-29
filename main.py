from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI()

# Root path to verify service is live
@app.get("/")
def read_root():
    return {"message": "YouTube Whisper API is running!"}

# Model to receive YouTube URL
class VideoURL(BaseModel):
    url: str

# Placeholder endpoint for transcription
@app.post("/transcribe")
async def transcribe(video: VideoURL):
    # Replace this section with actual transcription logic
    return {"status": "received", "video_url": video.url}

