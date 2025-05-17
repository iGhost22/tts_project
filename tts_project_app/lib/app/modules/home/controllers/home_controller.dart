import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../../../data/services/tts_service.dart';

class HomeController extends GetxController {
  final TtsService _ttsService = Get.find<TtsService>();
  
  // Text controller for the input field
  final textController = TextEditingController();
  
  // Giới hạn độ dài văn bản (ký tự)
  final int maxTextLength = 500;
  
  // Get reactive variables from the service
  RxBool get isLoading => _ttsService.isLoading;
  RxBool get isPlaying => _ttsService.isPlaying;
  RxString get errorMessage => _ttsService.errorMessage;
  RxString get audioFilePath => _ttsService.audioFilePath;
  
  // Biến theo dõi số ký tự
  RxInt get currentTextLength => RxInt(textController.text.length);
  RxBool get isTextTooLong => RxBool(textController.text.length > maxTextLength);
  
  // Kiểm tra kết nối API khi khởi động
  @override
  void onInit() {
    super.onInit();
    // Lắng nghe thay đổi văn bản để cập nhật số ký tự
    textController.addListener(() {
      update(); // Cập nhật UI khi văn bản thay đổi
    });
  }
  
  // Hàm kiểm tra kết nối API
  Future<void> checkApiConnection() async {
    try {
      if (textController.text.isEmpty) {
        textController.text = "Test connection";
      }
      await _ttsService.generateSpeech(textController.text);
    } catch (e) {
      errorMessage.value = e.toString();
      Get.snackbar(
        'Lỗi kết nối',
        'Không thể kết nối đến máy chủ TTS. Vui lòng kiểm tra API.',
        snackPosition: SnackPosition.TOP,
        backgroundColor: Colors.red[100],
        colorText: Colors.red[900],
        duration: const Duration(seconds: 5),
      );
    }
  }
  
  // Generate speech from text
  Future<void> generateSpeech() async {
    if (textController.text.isEmpty) {
      Get.snackbar(
        'Lỗi', 
        'Vui lòng nhập văn bản',
        snackPosition: SnackPosition.BOTTOM,
        backgroundColor: Colors.red,
        colorText: Colors.white,
      );
      return;
    }
    
    // Kiểm tra độ dài văn bản
    if (textController.text.length > maxTextLength) {
      Get.snackbar(
        'Văn bản quá dài', 
        'Vui lòng giữ văn bản dưới $maxTextLength ký tự. Hiện tại: ${textController.text.length} ký tự.',
        snackPosition: SnackPosition.BOTTOM,
        backgroundColor: Colors.orange,
        colorText: Colors.white,
        duration: const Duration(seconds: 5),
      );
      return;
    }
    
    try {
      await _ttsService.generateSpeech(textController.text);
      
      if (errorMessage.value.isNotEmpty) {
        Get.snackbar(
          'Lỗi', 
          errorMessage.value,
          snackPosition: SnackPosition.BOTTOM,
          backgroundColor: Colors.red,
          colorText: Colors.white,
          duration: const Duration(seconds: 5),
        );
      } else if (audioFilePath.value.isNotEmpty) {
        Get.snackbar(
          'Thành công', 
          'Đã tạo âm thanh thành công',
          snackPosition: SnackPosition.BOTTOM,
          backgroundColor: Colors.green,
          colorText: Colors.white,
        );
      }
    } catch (e) {
      Get.snackbar(
        'Lỗi xử lý', 
        e.toString(),
        snackPosition: SnackPosition.BOTTOM,
        backgroundColor: Colors.red,
        colorText: Colors.white,
        duration: const Duration(seconds: 5),
      );
    }
  }
  
  // Play generated audio
  void playAudio() {
    _ttsService.playAudio();
  }
  
  // Stop playing audio
  void stopAudio() {
    _ttsService.stopAudio();
  }
  
  // Clear text input
  void clearText() {
    textController.clear();
    update(); // Cập nhật UI sau khi xóa
  }
  
  @override
  void onClose() {
    textController.dispose();
    super.onClose();
  }
}
