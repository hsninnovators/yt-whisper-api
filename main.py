from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from faster_whisper import WhisperModel
import subprocess
import uuid
import os

app = FastAPI()

# Load faster-whisper model
model = WhisperModel("base", device="cpu", compute_type="int8")

# Pydantic input model
class TranscriptionRequest(BaseModel):
    url: str
    timestamps: bool = True
    download_txt: bool = False

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
async def transcribe(req: TranscriptionRequest):
    output_dir = "/tmp"
    try:
        audio_path = download_audio(req.url, output_dir)
        segments, info = model.transcribe(audio_path)

        language = info.language
        transcript = ""

        for segment in segments:
            text = segment.text.strip()
            if req.timestamps:
                start = round(segment.start, 2)
                end = round(segment.end, 2)
                transcript += f"[{start} - {end}] {text}\n"
            else:
                transcript += f"{text}\n"

        os.remove(audio_path)  # Clean up audio

        if req.download_txt:
            txt_file = os.path.join(output_dir, f"{uuid.uuid4()}.txt")
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(transcript.strip())
            return FileResponse(txt_file, filename="transcript.txt", media_type="text/plain")

        return {
            "language": language,
            "transcript": transcript.strip()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Whisper API is live. Use POST /transcribe"}
