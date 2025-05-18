# tts_api/app.py

import os
import sys
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import generate_speech

app = FastAPI(
    title="Text-to-Speech API",
    description="API for text-to-speech using Tacotron",
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origins (trong môi trường sản phẩm nên giới hạn)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đường dẫn cố định đến file âm thanh
AUDIO_FILE_PATH = os.path.join("result", "output.wav")


class TextToSpeechRequest(BaseModel):
    text: str


@app.get("/")
def read_root():
    """
    Root endpoint to check API status
    """
    return {"status": "online", "message": "Text-to-Speech API is running"}


@app.post("/tts")
def text_to_speech(request: TextToSpeechRequest):
    """
    Convert text to speech and return the audio file
    """
    try:
        # Chạy mô hình và lấy đường dẫn file
        audio_path = generate_speech(request.text)

        # Kiểm tra xem file có tồn tại không
        if not os.path.exists(audio_path):
            raise Exception(f"Audio file not created at {audio_path}")

        # Trả về file
        return FileResponse(
            path=audio_path,
            media_type="audio/wav",
            filename="output.wav",
            content_disposition_type="inline",  # Cho phép nghe trực tiếp
        )
    except Exception as e:
        error_message = str(e)
        # Log lỗi chi tiết
        import traceback
        print(f"Error in /tts endpoint: {error_message}")
        print(traceback.format_exc())
        
        # Trả về lỗi chi tiết hơn
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating speech: {error_message}"
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
