# tts_api/app.py

import os
import sys
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import generate_speech

app = FastAPI(
    title="Text-to-Speech API",
    description="API for text-to-speech using Tacotron",
)

# Đường dẫn cố định đến file âm thanh
AUDIO_FILE_PATH = os.path.join("result", "output.wav")


class TextToSpeechRequest(BaseModel):
    text: str


# @app.get("/")
# def read_root():
#     return {"message": "Text-to-Speech API"}


@app.post("/tts")
def text_to_speech(request: TextToSpeechRequest):
    """
    Convert text to speech and return the audio file
    """
    try:
        # Chạy mô hình và lấy đường dẫn file
        audio_path = generate_speech(request.text)

        # Trả về file
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename="output.wav",
            content_disposition_type="inline",  # Cho phép nghe trực tiếp
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating speech: {str(e)}"
        )


# @app.get("/tts")
# def text_to_speech_get(text: str = Query(..., description="Text to convert to speech")):
#     """
#     Convert text to speech using GET method and return the audio file
#     """
#     try:
#         # Generate a unique filename
#         filename = f"speech_{uuid.uuid4().hex[:8]}"

#         # Generate the speech
#         audio_path = generate_speech(text, filename)

#         # Return the audio file
#         return FileResponse(
#             path=audio_path, media_type="audio/wav", filename=f"{filename}.wav"
#         )
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Error generating speech: {str(e)}"
#         )


@app.get("/audio")
def get_audio():
    """
    Get the generated audio file
    """
    if not os.path.exists(AUDIO_FILE_PATH):
        raise HTTPException(
            status_code=404, detail="Audio file not found. Generate speech first."
        )
    return FileResponse(
        path=AUDIO_FILE_PATH,
        media_type="audio/wav",
        filename="output.wav",
        content_disposition_type="inline",  # Cho phép nghe trực tiếp
    )


# @app.get("/audio")
# def get_audio():
#     """
#     Get the generated audio file
#     """
#     if not os.path.exists(AUDIO_FILE_PATH):
#         raise HTTPException(
#             status_code=404, detail="Audio file not found. Generate speech first."
#         )
#     return FileResponse(
#         path=AUDIO_FILE_PATH,
#         media_type="audio/wav",
#         filename="output.wav",
#         content_disposition_type="inline",  # Cho phép nghe trực tiếp
#     )
