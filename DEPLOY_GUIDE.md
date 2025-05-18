# Hướng dẫn triển khai TTS API

## Chuẩn bị

1. **Đăng ký tài khoản GitHub** (nếu chưa có)
2. **Fork repository này** vào tài khoản GitHub của bạn

## Triển khai lên Railway

### Phương pháp 1: Triển khai tự động qua GitHub

1. **Đăng ký tài khoản Railway**
   - Truy cập [Railway.app](https://railway.app)
   - Đăng ký tài khoản (có thể đăng nhập bằng GitHub)

2. **Tạo dự án mới trong Railway**
   - Nhấp vào "New Project"
   - Chọn "Deploy from GitHub repo"
   - Cho phép Railway truy cập repository của bạn
   - Chọn repository đã fork

3. **Cấu hình triển khai**
   - Railway sẽ tự động phát hiện Dockerfile
   - Đợi quá trình build và deploy hoàn tất (khoảng 5-10 phút)

4. **Tạo domain công khai**
   - Trong dự án, nhấp vào tab "Settings"
   - Nhấp vào "Generate Domain"
   - Railway sẽ tạo một URL công khai cho API của bạn

5. **Kiểm tra API**
   - Truy cập đường dẫn được tạo để kiểm tra API
   - Thử endpoint: `https://your-app.up.railway.app/`
   - Kiểm tra docs: `https://your-app.up.railway.app/docs`

### Phương pháp 2: Sử dụng Railway CLI

1. **Cài đặt Railway CLI**
   ```bash
   npm install -g @railway/cli
   ```

2. **Đăng nhập vào Railway từ CLI**
   ```bash
   railway login
   ```

3. **Khởi tạo dự án**
   ```bash
   cd /đường/dẫn/đến/repository
   railway init
   ```

4. **Triển khai ứng dụng**
   ```bash
   railway up
   ```

5. **Tạo domain công khai**
   ```bash
   railway domain
   ```

## Quản lý và giám sát

1. **Xem logs**
   - Trong dashboard Railway, nhấp vào "Deployments"
   - Chọn deployment hiện tại
   - Xem logs để kiểm tra lỗi

2. **Restart ứng dụng nếu cần**
   - Trong dashboard, nhấp vào "Deployments"
   - Nhấp vào "..." bên cạnh deployment
   - Chọn "Restart"

## Sử dụng API

1. **Chuyển văn bản thành giọng nói**
   ```bash
   curl -X POST "https://your-app.up.railway.app/tts" \
     -H "Content-Type: application/json" \
     -d '{"text":"Xin chào, tôi là trợ lý ảo."}'
   ```

2. **Lấy file âm thanh đã tạo**
   ```bash
   curl -X GET "https://your-app.up.railway.app/audio"
   ```

## Xử lý sự cố

- **Lỗi "Memory limit exceeded"**: Dự án có thể cần nhiều RAM hơn. Xem xét nâng cấp gói Railway từ free lên paid.
- **Lỗi thời gian chạy quá lâu**: Quá trình tạo âm thanh có thể mất nhiều thời gian với văn bản dài. Thử với văn bản ngắn hơn.
- **Lỗi mô hình không tải**: Kiểm tra logs để xác nhận đường dẫn `CHECKPOINT_PATH` đúng và file checkpoint đã được sao chép.

## Trích xuất URL và tích hợp

Sau khi triển khai, bạn sẽ có một URL dạng:
```
https://your-app-name.up.railway.app
```

Bạn có thể sử dụng URL này để:
1. Truy cập trực tiếp API
2. Tích hợp vào ứng dụng web hoặc mobile
3. Chia sẻ với người khác để họ có thể sử dụng API của bạn 