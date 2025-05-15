# tts_api/inference.py

import os
import sys
import numpy as np
import soundfile as sf
import torch
from torch.autograd import Variable

# Cho phép import module từ thư mục gốc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.tacotron import Tacotron
from utils import audio
from utils.text import text_to_sequence, symbols
from utils.plot import test_visualize
from config import config

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ===== Load model =====
def load_model(checkpoint_path="../ckpt/checkpoint_step500000.pth"):
	print(f"[INFO] Loading model from {checkpoint_path}")
	model = Tacotron(
		n_vocab=len(symbols),
		embedding_dim=config.embedding_dim,
		mel_dim=config.num_mels,
		linear_dim=config.num_freq,
		r=config.outputs_per_step,
		padding_idx=config.padding_idx,
		attention=config.attention,
		use_mask=config.use_mask
	)
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint["state_dict"])
	model.decoder.max_decoder_steps = config.max_decoder_steps
	model.eval()
	return model.to(device)

model = load_model()

# ===== TTS core function (from your test.py) =====
def tts(model, text):
	model.encoder.eval()
	model.postnet.eval()

	sequence = np.array(text_to_sequence(text))
	if sequence.shape[0] == 0:
		raise ValueError("Text could not be converted to sequence. Please try a different input.")

	sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0).to(device)
	with torch.no_grad():
		mel_outputs, linear_outputs, gate_outputs, alignments = model(sequence)

	linear_output = linear_outputs[0].cpu().data.numpy()
	spectrogram = audio._denormalize(linear_output)
	alignment = alignments[0].cpu().data.numpy()
	waveform = audio.inv_spectrogram(linear_output.T)
	return waveform, alignment, spectrogram

# ===== Public API for FastAPI =====
def synthesize(text: str, output_path: str = "result/output.wav", plot: bool = True):
    
	if not text.strip():
		raise ValueError("Input text must not be empty.")

	print(f"[INFO] Synthesizing: \"{text}\"")
	waveform, alignment, spectrogram = tts(model, text)

	# Tạo thư mục nếu chưa có
	os.makedirs(os.path.dirname(output_path), exist_ok=True)

	# Ghi file wav
	sf.write(output_path, waveform, config.sample_rate)

	# Ghi hình ảnh alignment và spectrogram nếu cần
	if plot:
		test_visualize(alignment, spectrogram, output_path)

	return output_path
