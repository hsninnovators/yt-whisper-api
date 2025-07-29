from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import subprocess
import os
import uuid
import whisper

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whisper model load (use small, medium or large if needed)
model = whisper.load_model("base")

# Base route to test API is running
@app.get("/")
async def root():
    return {"message": "YouTube Whisper API is running!"}

# Request body schema
class TranscriptRequest(BaseModel):
    url: str

# Transcription endpoint
@app.post("/transcribe")
async def transcribe(req: TranscriptRequest):
    try:
        video_url = req.url
        audio_filename = f"audio_{uuid.uuid4()}.mp3"

        # Use yt-dlp to download audio
        command = [
            "yt-dlp",
            "-x", "--audio-format", "mp3",
            "--output", audio_filename,
            video_url
        ]
        subprocess.run(command, check=True)

        # Transcribe using whisper
        result = model.transcribe(audio_filename)

        # Clean up the file
        os.remove(audio_filename)

        return {"text": result["text"]}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail="Failed to download audio.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


