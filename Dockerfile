   FROM python:3.9-slim

   WORKDIR /app

   # Cài đặt dependencies hệ thống
   RUN apt-get update && apt-get install -y --no-install-recommends \
       build-essential \
       libsndfile1 \
       && rm -rf /var/lib/apt/lists/*
       
   # Copy và đổi tên file requirements
   COPY tts_api/requirements_render.txt ./requirements.txt
   
   # Chia quá trình cài đặt để dễ debug
   RUN pip install --no-cache-dir -U pip setuptools wheel
   RUN pip install --no-cache-dir numpy==1.24.3
   RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
   RUN pip install --no-cache-dir -r requirements.txt
   
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

   # Lệnh chạy
   CMD cd tts_api && python -c "import numpy as np; print(f'NumPy version: {np.__version__}')" && uvicorn app:app --host 0.0.0.0 --port ${PORT}