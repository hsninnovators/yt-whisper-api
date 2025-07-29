from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os

app = FastAPI()

class VideoURL(BaseModel):
    url: str

@app.post("/transcribe/")
async def transcribe_video(video: VideoURL):
    try:
        response = requests.post(
            "https://hsn-whisper.hf.space/run/predict",
            json={"data": [video.url]},
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        return {"transcription": result["data"][0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
