from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from faster_whisper import WhisperModel
import subprocess
import uuid
import os

app = FastAPI()

# Load lightweight Whisper model
model = WhisperModel("base", device="cpu", compute_type="int8")

# Input model
class VideoURL(BaseModel):
    url: str

def download_audio(url: str, output_path: str) -> str:
    audio_file = os.path.join(output_path, f"{uuid.uuid4()}.mp3")
    command = [
        "yt-dlp",
        "-f", "bestaudio",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "-o", audio_file,
        url
    ]
    try:
        subprocess.run(command, check=True)
        return audio_file
    except subprocess.CalledProcessError as e:
        raise Exception(f"yt-dlp failed: {str(e)}")

@app.post("/transcribe")
async def transcribe(video: VideoURL):
    output_dir = "/tmp"
    try:
        audio_path = download_audio(video.url, output_dir)
        segments, _ = model.transcribe(audio_path)

        transcript = ""
        for segment in segments:
            start = round(segment.start, 2)
            end = round(segment.end, 2)
            text = segment.text.strip()
            transcript += f"[{start} - {end}] {text}\n"

        os.remove(audio_path)  # cleanup
        return {"transcript": transcript.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Whisper transcription API is live."}
