import 'package:flutter/material.dart';

import 'package:get/get.dart';

import '../controllers/home_controller.dart';

class HomeView extends GetView<HomeController> {
  const HomeView({super.key});
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Text to Speech'),
        centerTitle: true,
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Text input area
            GetBuilder<HomeController>(
              builder: (_) => Column(
                crossAxisAlignment: CrossAxisAlignment.end,
                children: [
                  TextField(
                    controller: controller.textController,
                    maxLines: 5,
                    decoration: InputDecoration(
                      hintText: 'Input text here...',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                      fillColor: Colors.grey[100],
                      filled: true,
                      contentPadding: const EdgeInsets.all(16),
                    ),
                  ),
                  const SizedBox(height: 4),
                  // Hiển thị số ký tự / giới hạn
                  Text(
                    '${controller.textController.text.length}/${controller.maxTextLength} ký tự',
                    style: TextStyle(
                      color: controller.textController.text.length >
                              controller.maxTextLength
                          ? Colors.red
                          : Colors.grey[600],
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),

            // Button row for actions
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: controller.clearText,
                    icon: const Icon(Icons.clear),
                    label: const Text('Clear'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.grey,
                      foregroundColor: Colors.white,
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Obx(() => ElevatedButton.icon(
                        onPressed: controller.isLoading.value ||
                                controller.textController.text.length >
                                    controller.maxTextLength
                            ? null
                            : controller.generateSpeech,
                        icon: controller.isLoading.value
                            ? const SizedBox(
                                width: 20,
                                height: 20,
                                child:
                                    CircularProgressIndicator(strokeWidth: 2))
                            : const Icon(Icons.record_voice_over),
                        label: Text(controller.isLoading.value
                            ? 'Generating...'
                            : 'Generate'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.blue,
                          foregroundColor: Colors.white,
                          disabledBackgroundColor: Colors.blue.withOpacity(0.5),
                        ),
                      )),
                ),
              ],
            ),

            const SizedBox(height: 24),

            // Audio player controls
            Obx(() => Visibility(
                  visible: controller.audioFilePath.value.isNotEmpty,
                  child: Card(
                    elevation: 4,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Audio Ready',
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          const SizedBox(height: 16),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              ElevatedButton.icon(
                                onPressed: controller.isPlaying.value
                                    ? controller.stopAudio
                                    : controller.playAudio,
                                icon: Icon(
                                  controller.isPlaying.value
                                      ? Icons.stop
                                      : Icons.play_arrow,
                                ),
                                label: Text(
                                  controller.isPlaying.value ? 'Stop' : 'Play',
                                ),
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: controller.isPlaying.value
                                      ? Colors.red
                                      : Colors.green,
                                  foregroundColor: Colors.white,
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(20),
                                  ),
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 24, vertical: 12),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),
                )),

            // Error message
            Obx(() => Visibility(
                  visible: controller.errorMessage.value.isNotEmpty,
                  child: Padding(
                    padding: const EdgeInsets.only(top: 16.0),
                    child: Card(
                      color: Colors.red[50],
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                        side: BorderSide(color: Colors.red.shade200),
                      ),
                      child: Padding(
                        padding: const EdgeInsets.all(12.0),
                        child: Text(
                          controller.errorMessage.value,
                          style: TextStyle(
                            color: Colors.red[900],
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ),
                    ),
                  ),
                )),
          ],
        ),
      ),
    );
  }
}
