FROM python:3.9-slim

WORKDIR /app

# Cài đặt dependencies hệ thống
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt dependencies Python
COPY tts_api/requirements_render.txt .
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements_render.txt

# Tải dữ liệu NLTK
RUN python -m nltk.downloader punkt cmudict

# Sao chép mã nguồn
COPY config.py .
COPY model/ ./model/
COPY utils/ ./utils/
COPY ckpt/checkpoint_step500000.pth ./ckpt/checkpoint_step500000.pth
COPY tts_api/ ./tts_api/

# Tạo thư mục result
RUN mkdir -p result

# Thiết lập biến môi trường
ENV CHECKPOINT_PATH=/app/ckpt/checkpoint_step500000.pth
ENV USE_CUDA=0
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Kiểm tra xem cấu trúc thư mục đã đúng chưa
RUN ls -la && \
    ls -la model && \
    ls -la utils && \
    ls -la ckpt && \
    ls -la tts_api

# Lệnh chạy
CMD cd tts_api && python -c "import numpy as np; print(f'NumPy version: {np.__version__}')" && uvicorn app:app --host 0.0.0.0 --port ${PORT} 