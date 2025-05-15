# tts_api/app.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from inference import synthesize

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/synthesize")
def generate_audio(input: TextInput):
    try:
        output_path = "result/output.wav"
        synthesize(input.text, output_path=output_path, plot=True)
        return {"audio_url": f"/audio/output.wav"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/audio/{filename}")
def get_audio(filename: str):
    file_path = os.path.join("result", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="audio/wav")
