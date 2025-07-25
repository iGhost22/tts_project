import 'dart:io';

import 'package:audioplayers/audioplayers.dart';
import 'package:get/get.dart';

import '../models/tts_request_model.dart';
import '../providers/tts_provider.dart';

class TtsService extends GetxService {
  final TtsProvider _ttsProvider = TtsProvider();
  final AudioPlayer _audioPlayer = AudioPlayer();

  final RxBool isLoading = false.obs;
  final RxBool isPlaying = false.obs;
  final RxString audioFilePath = ''.obs;
  final RxString errorMessage = ''.obs;
  final RxDouble playbackProgress = 0.0.obs;
  Duration? _duration;

  Future<String> generateSpeech(String text) async {
    try {
      isLoading.value = true;
      errorMessage.value = '';

      final request = TtsRequest(text: text);
      final filePath = await _ttsProvider.generateSpeech(request);

      audioFilePath.value = filePath;
      return filePath;
    } catch (e) {
      errorMessage.value = e.toString();
      return '';
    } finally {
      isLoading.value = false;
    }
  }

  @override
  void onClose() {
    _audioPlayer.dispose();
    super.onClose();
  }

  Future<void> playAudio() async {
    try {
      if (audioFilePath.value.isEmpty) {
        errorMessage.value = 'No audio file to play';
        return;
      }

      // Kiểm tra file tồn tại
      final file = File(audioFilePath.value);
      if (!await file.exists()) {
        errorMessage.value = 'Audio file not found';
        return;
      }

      // Dừng phát âm thanh hiện tại nếu có
      await _audioPlayer.stop();

      // Đặt lại trạng thái
      isPlaying.value = false;
      playbackProgress.value = 0.0;
      _duration = null;

      // Cấu hình AudioPlayer
      await _audioPlayer.setSource(DeviceFileSource(audioFilePath.value));

      // Lắng nghe sự kiện phát âm thanh
      _audioPlayer.onPositionChanged.listen((Duration position) {
        if (_duration != null) {
          playbackProgress.value = position.inMilliseconds / _duration!.inMilliseconds;
        }
      });

      _audioPlayer.onDurationChanged.listen((Duration duration) {
        _duration = duration;
      });

      _audioPlayer.onPlayerComplete.listen((event) {
        isPlaying.value = false;
        playbackProgress.value = 1.0;
      });

      // Bắt đầu phát âm thanh
      await _audioPlayer.resume();
      isPlaying.value = true;
    } catch (e) {
      errorMessage.value = 'Error playing audio: $e';
      isPlaying.value = false;
    }
  }

  Future<void> stopAudio() async {
    try {
      await _audioPlayer.stop();
      isPlaying.value = false;
      playbackProgress.value = 0.0;
      _duration = null;
    } catch (e) {
      errorMessage.value = 'Error stopping audio: $e';
    }
  }
}
