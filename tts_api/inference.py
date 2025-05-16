import os
import sys
import torch
from torch.autograd import Variable
import numpy as np
import soundfile as sf
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from project modules
from utils import audio
from utils.text import text_to_sequence, symbols
from utils.plot import test_visualize
from model.tacotron import Tacotron
from config import config

USE_CUDA = torch.cuda.is_available()


class TTSModel:
    def __init__(self, checkpoint_path="../ckpt/checkpoint_step500000.pth"):
        self.model = None
        self.checkpoint_path = checkpoint_path
        self.result_dir = "result"
        self.output_file = "output.wav"
        os.makedirs(self.result_dir, exist_ok=True)
        self.load_model()

    def load_model(self):
        print(f"Loading model from {self.checkpoint_path}")
        self.model = Tacotron(
            n_vocab=len(symbols),
            embedding_dim=config.embedding_dim,
            mel_dim=config.num_mels,
            linear_dim=config.num_freq,
            r=config.outputs_per_step,
            padding_idx=config.padding_idx,
            attention=config.attention,
            use_mask=config.use_mask,
        )

        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )

        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.decoder.max_decoder_steps = config.max_decoder_steps * 2

        if USE_CUDA:
            self.model = self.model.cuda()

        self.model.encoder.eval()
        self.model.postnet.eval()

    def tts(self, text):
        """
        Chuyển văn bản thành giọng nói với chất lượng cao

        Args:
            text: Văn bản đầu vào

        Returns:
            waveform: Dạng sóng âm thanh
            alignment: Ma trận alignment
            spectrogram: Spectrogram
        """
        if not text or not text.strip():
            raise ValueError("Văn bản đầu vào không được để trống")

        sequence = np.array(text_to_sequence(text))
        sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)

        if USE_CUDA:
            sequence = sequence.cuda()

        mel_outputs, linear_outputs, gate_outputs, alignments = self.model(sequence)
        linear_output = linear_outputs[0].cpu().data.numpy()
        spectrogram = audio._denormalize(linear_output)
        alignment = alignments[0].cpu().data.numpy()

        waveform = audio.inv_spectrogram(linear_output.T)

        return waveform, alignment, spectrogram

    def synthesize_speech(
        self, text, output_filename=None, generate_figures=True, enhance_audio=True
    ):
        """
        Tổng hợp âm thanh từ văn bản và lưu vào file với chất lượng cao

        Args:
            text: Văn bản đầu vào
            output_filename: Bỏ qua, luôn sử dụng tên file cố định
            generate_figures: Tạo hình ảnh alignment và spectrogram
            enhance_audio: Áp dụng cải tiến chất lượng âm thanh

        Returns:
            Path to the generated audio file
        """
        output_path = os.path.join(self.result_dir, self.output_file)
        output_without_ext = (
            output_path.rsplit(".", 1)[0] if "." in output_path else output_path
        )

        print(f"Synthesizing: '{text}'")
        try:
            waveform, alignment, spectrogram = self.tts(text)

            if generate_figures:
                figure_path = os.path.join(
                    self.result_dir, f"figure_{int(time.time())}"
                )
                test_visualize(alignment, spectrogram, figure_path)

            if enhance_audio:
                waveform = waveform * (0.95 / max(0.01, np.max(np.abs(waveform))))

            sf.write(
                output_without_ext + ".wav",
                waveform,
                config.sample_rate,
                subtype="PCM_24",
            )

            print(f"Đã lưu âm thanh: {output_without_ext}.wav")
            return output_without_ext + ".wav"

        except Exception as e:
            print(f"Lỗi khi tổng hợp âm thanh: {str(e)}")
            raise


tts_model = TTSModel()


def generate_speech(text, filename=None):
    """
    Generate speech from text and return the path to the audio file

    Args:
        text: Text to synthesize
        filename: Ignored, always uses fixed output filename

    Returns:
        Path to the generated audio file
    """
    return tts_model.synthesize_speech(text, generate_figures=True, enhance_audio=True)
