# tts_api/app.py

import os
import sys

# Thêm thư mục hiện tại vào đường dẫn Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Đảm bảo USE_CUDA=0 để tránh vấn đề về CUDA/GPU
os.environ["USE_CUDA"] = "0"

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid

try:
    from inference import generate_speech
except ImportError as e:
    print(f"Error importing inference module: {e}")
    # Tiếp tục thực thi, xử lý lỗi trong endpoint

app = FastAPI(
    title="Text-to-Speech API",
    description="API for text-to-speech using Tacotron",
)

# Đường dẫn cố định đến file âm thanh
AUDIO_FILE_PATH = os.path.join("result", "output.wav")


class TextToSpeechRequest(BaseModel):
    text: str


@app.get("/")
def read_root():
    return {"message": "Text-to-Speech API", "status": "online"}


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
        print(f"Error in text_to_speech: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating speech: {str(e)}"
        )


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


@app.get("/health")
def health_check():
    """
    Simple health check endpoint
    """
    return {"status": "healthy"}
