import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

import '../models/tts_request_model.dart';

class TtsProvider {
  // final String baseUrl = 'http://localhost:8000'; // Sử dụng cho iOS Simulator để kết nối tới localhost của máy host

  final String baseUrl = Platform.isAndroid
      ? 'http://10.0.2.2:8000' // Cho Android Emulator
      : 'http://localhost:8000'; // Cho iOS Simulator

  // Tăng timeout lên 30 giây để xử lý văn bản dài
  final Duration requestTimeout = const Duration(seconds: 30);

  Future<String> generateSpeech(TtsRequest request) async {
    try {
      // Kiểm tra độ dài văn bản
      if (request.text.length > 500) {
        // write by english
        throw Exception('Text too long: The text exceeds the maximum length of 500 characters.');
      }

      final response = await http
          .post(
            Uri.parse('$baseUrl/tts'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(request.toJson()),
          )
          .timeout(requestTimeout);

      if (response.statusCode == 200) {
        // Tạo tên file duy nhất dựa trên timestamp
        final timestamp = DateTime.now().millisecondsSinceEpoch;
        final directory = await getTemporaryDirectory();
        final file = File('${directory.path}/output_$timestamp.wav');

        // Lưu file âm thanh
        await file.writeAsBytes(response.bodyBytes);

        // Kiểm tra kích thước file
        final fileSize = await file.length();
        if (fileSize == 0) {
          throw Exception('File âm thanh trống');
        }

        return file.path;
      } else {
        throw Exception('Lỗi API: Mã ${response.statusCode}. ${response.body}');
      }
    } on SocketException catch (e) {
      throw Exception('Lỗi kết nối: Không thể kết nối đến máy chủ. Vui lòng đảm bảo API đang chạy và URL chính xác.');
    } on HttpException catch (e) {
      throw Exception('Lỗi HTTP: $e');
    } on FormatException catch (e) {
      throw Exception('Lỗi định dạng dữ liệu: $e');
    } on TimeoutException catch (e) {
      throw Exception('Văn bản quá dài hoặc máy chủ phản hồi chậm. Vui lòng thử với văn bản ngắn hơn.');
    } catch (e) {
      throw Exception('Lỗi xử lý: $e');
    }
  }
}
