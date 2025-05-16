import os
import sys
import nltk
import librosa
import librosa.display
import numpy as np
from tqdm import tqdm
import soundfile as sf

# ------------------------#
import torch
from torch.autograd import Variable

from utils import audio
from utils.text import text_to_sequence, symbols
from utils.plot import test_visualize, plot_alignment
from model.tacotron import Tacotron
from config import config

USE_CUDA = torch.cuda.is_available()


# ============ TTS Core ==============
def tts(model, text):
    if USE_CUDA:
        model = model.cuda()
    model.encoder.eval()
    model.postnet.eval()

    sequence = np.array(text_to_sequence(text))
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
    if USE_CUDA:
        sequence = sequence.cuda()

    mel_outputs, linear_outputs, gate_outputs, alignments = model(sequence)
    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio._denormalize(linear_output)
    alignment = alignments[0].cpu().data.numpy()
    waveform = audio.inv_spectrogram(linear_output.T)
    return waveform, alignment, spectrogram


# ============ Save Audio ==============
def synthesis_speech(model, text, figures=True, path=None):
    waveform, alignment, spectrogram = tts(model, text)
    if figures:
        test_visualize(alignment, spectrogram, path)
    sf.write(path + ".wav", waveform, config.sample_rate)


# ============ Main ==============
def main():
    # Cấu hình cố định
    ckpt_dir = "ckpt/"
    model_name = "500000"
    result_dir = "result/"
    checkpoint_path = os.path.join(ckpt_dir, "checkpoint_step" + model_name + ".pth")
    output_name = os.path.join(result_dir, model_name)

    # Load model
    os.makedirs(result_dir, exist_ok=True)
    print(f"Loading model from {checkpoint_path}")
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.decoder.max_decoder_steps = config.max_decoder_steps

    # Interactive loop
    while True:
        try:
            text = input("< Tacotron > Text to speech: ")
            print("Model input:", text)
            synthesis_speech(model, text=text, figures=True, path=output_name)
            print(f"[✓] Output saved to: {output_name}.wav")
        except KeyboardInterrupt:
            print("\n[!] Terminated by user.")
            break


if __name__ == "__main__":
    main()
