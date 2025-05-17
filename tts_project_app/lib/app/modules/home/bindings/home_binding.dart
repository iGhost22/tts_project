import 'package:get/get.dart';
import '../../../data/services/tts_service.dart';
import '../controllers/home_controller.dart';

class HomeBinding extends Bindings {
  @override
  void dependencies() {
    // Register the TTS service
    Get.put<TtsService>(TtsService(), permanent: true);
    
    // Register the home controller
    Get.lazyPut<HomeController>(
      () => HomeController(),
    );
  }
}
