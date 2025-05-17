# Text-to-Speech App

Ứng dụng Flutter kết nối với API Text-to-Speech để chuyển đổi văn bản thành giọng nói.

## Cài đặt và Chạy ứng dụng

### Bước 1: Chạy API TTS
API cần được chạy trước khi sử dụng app Flutter. Di chuyển vào thư mục API:

```bash
cd ../tts_api
```

Cài đặt các dependencies:
```bash
pip install -r requirements.txt
```

Chạy API:
```bash
python run.py
```

API sẽ chạy tại địa chỉ http://localhost:8000

### Bước 2: Cấu hình kết nối API trong Flutter

Mở file `lib/app/data/providers/tts_provider.dart` và điều chỉnh `baseUrl` phù hợp với cách bạn chạy:

- Chạy trên máy ảo Android: Sử dụng `10.0.2.2` (mặc định hiện tại)
- Chạy trên thiết bị thực: Sử dụng địa chỉ IP của máy tính chạy API (ví dụ: `192.168.1.5`)

### Bước 3: Chạy app Flutter

```bash
flutter pub get
flutter run
```

## Triển khai API miễn phí

Để tránh phải chạy API cục bộ, bạn có thể triển khai API lên các dịch vụ hosting miễn phí:

1. **Render**: Cung cấp dịch vụ hosting miễn phí cho web services
   - Đăng ký tại https://render.com/
   - Kết nối với repo GitHub của bạn
   - Cấu hình build command: `pip install -r requirements.txt`
   - Start command: `cd tts_api && uvicorn app:app --host 0.0.0.0 --port $PORT`

2. **Railway**: Nền tảng triển khai ứng dụng đơn giản với gói miễn phí
   - Đăng ký tại https://railway.app/
   - Kết nối với repo GitHub
   - Cấu hình tự động sẽ được thiết lập

3. **Fly.io**: Cung cấp các máy ảo nhỏ với gói miễn phí
   - Đăng ký tại https://fly.io/
   - Sử dụng Flyctl để triển khai

Sau khi triển khai, cập nhật `baseUrl` trong Flutter app với URL của API đã triển khai.

## Xử lý lỗi kết nối

Nếu gặp lỗi kết nối, hãy kiểm tra:
- API đang chạy
- Địa chỉ IP/hostname chính xác
- Firewall không chặn kết nối
- Nếu chạy trên thiết bị thực, đảm bảo cả thiết bị và máy chủ API cùng mạng
