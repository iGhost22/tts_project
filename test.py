import os
import sys
import nltk
import argparse
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
import soundfile as sf

# --------------------------------#
import torch
from torch.autograd import Variable

# --------------------------------#
from utils import audio
from utils.text import text_to_sequence, symbols
from utils.plot import test_visualize, plot_alignment

# --------------------------------#
from model.tacotron import Tacotron
from config import config, get_test_args


# CONSTANT #
USE_CUDA = torch.cuda.is_available()


# TEXT TO SPEECH #
def tts(model, text):
    """Convert text to speech waveform given a Tacotron model with improved quality."""
    if USE_CUDA:
        model = model.cuda()

    # Đảm bảo model ở chế độ evaluation
    model.encoder.eval()
    model.postnet.eval()

    # Chuyển đổi text thành sequence
    sequence = np.array(text_to_sequence(text))
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
    if USE_CUDA:
        sequence = sequence.cuda()

    # Greedy decoding với attention cao hơn
    mel_outputs, linear_outputs, gate_outputs, alignments = model(sequence)

    # Lấy dữ liệu spectrogram
    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio._denormalize(linear_output)
    alignment = alignments[0].cpu().data.numpy()

    # Chuyển spectrogram thành waveform với chất lượng cao hơn
    waveform = audio.inv_spectrogram(linear_output.T)

    # Trả về kết quả đã xử lý
    return waveform, alignment, spectrogram


# SYNTHESIS SPEECH #
def synthesis_speech(model, text, figures=True, path=None, enhance_audio=True):
    """Tổng hợp âm thanh từ văn bản và lưu vào file với chất lượng cao.

    Args:
        model: Model Tacotron
        text: Văn bản đầu vào
        figures: True nếu muốn tạo hình ảnh alignment và spectrogram
        path: Đường dẫn lưu file (không có đuôi .wav)
        enhance_audio: Áp dụng các cải tiến chất lượng âm thanh
    """
    print(f"Đang tổng hợp: '{text}'")

    # Tạo waveform
    waveform, alignment, spectrogram = tts(model, text)

    # Tạo hình ảnh nếu cần
    if figures:
        test_visualize(alignment, spectrogram, path)

    # Chuẩn hóa và tăng cường âm thanh nếu yêu cầu
    if enhance_audio:
        # Tăng cường độ sáng của âm thanh bằng cách
        # đảm bảo giá trị peak gần với ngưỡng tối đa để âm thanh to và rõ hơn
        waveform = waveform * (0.95 / max(0.01, np.max(np.abs(waveform))))

    # Lưu file âm thanh với định dạng chất lượng cao
    sf.write(path + ".wav", waveform, config.sample_rate, subtype="PCM_24")

    print(f"Đã lưu âm thanh vào: {path}.wav")
    return path + ".wav"


# MAIN #
def main():

    # ---initialize---#
    args = get_test_args()

    # Cấu hình model Tacotron
    model = Tacotron(
        n_vocab=len(symbols),
        embedding_dim=config.embedding_dim,
        mel_dim=config.num_mels,
        linear_dim=config.num_freq,
        r=config.outputs_per_step,
        padding_idx=config.padding_idx,
        attention=config.attention,
        use_mask=config.use_mask,
    )

    # ---handle path---#
    checkpoint_path = os.path.join(
        args.ckpt_dir, args.checkpoint_name + args.model_name + ".pth"
    )
    os.makedirs(args.result_dir, exist_ok=True)

    # ---load and set model---#
    print("Loading model: ", checkpoint_path)

    # Đảm bảo load model đúng cách
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])

    # Tăng max_decoder_steps để xử lý văn bản dài tốt hơn
    model.decoder.max_decoder_steps = config.max_decoder_steps * 2

    if args.interactive == True:
        output_name = os.path.join(args.result_dir, args.model_name)

        # ---testing loop---#
        while True:
            try:
                text = str(input("< Tacotron > Text to speech: "))
                if not text.strip():
                    print("Văn bản không được để trống. Vui lòng thử lại.")
                    continue

                print("Model input: ", text)
                audio_path = synthesis_speech(
                    model, text=text, figures=args.plot, path=output_name
                )
                print(f"[✓] Đã lưu kết quả tại: {audio_path}")
            except KeyboardInterrupt:
                print()
                print("Kết thúc chương trình!")
                break
            except Exception as e:
                print(f"[!] Lỗi: {str(e)}")
                print("Vui lòng thử lại với văn bản khác.")

    elif args.interactive == False:
        output_dir = os.path.join(args.result_dir, args.model_name + "/")
        os.makedirs(output_dir, exist_ok=True)

        # ---testing flow---#
        with open(args.test_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            valid_lines = [line.strip() for line in lines if line.strip()]
            print(f"Tổng số mẫu cần xử lý: {len(valid_lines)}")

            for idx, line in enumerate(tqdm(valid_lines, desc="Đang xử lý")):
                try:
                    output_path = os.path.join(output_dir, f"sample_{idx+1}")
                    print(
                        f"[{idx+1}/{len(valid_lines)}]: '{line}' - ({len(line)} ký tự)"
                    )
                    synthesis_speech(
                        model, text=line, figures=args.plot, path=output_path
                    )
                except Exception as e:
                    print(f"[!] Lỗi khi xử lý mẫu {idx+1}: {str(e)}")
                    continue

        print(f"Hoàn thành! Kết quả được lưu tại: {output_dir}")

    else:
        raise RuntimeError("Chế độ không hợp lệ!")

    sys.exit(0)


if __name__ == "__main__":
    main()
