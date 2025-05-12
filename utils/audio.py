import math
import numpy as np
import librosa
import librosa.filters
from scipy import signal
from config import config
from scipy.io import wavfile

# FUNCTIONS #
"""
	Audio processing functions for loading, saving, and manipulating audio data.
	These functions are used to convert audio files to spectrograms and vice versa.
	They also include functions for pre-emphasizing audio, finding endpoints, and converting between linear and mel spectrograms.
	These functions are used in the preprocessing of audio data for training and inference in a speech synthesis model.
	Args:
		- wav: The audio waveform to be processed.
"""

# Load and save audio files
def load_wav(path):
	return librosa.core.load(path, sr=config.sample_rate)[0]

# Save audio files
# This function takes a waveform and a file path as input and saves the waveform as a WAV file at the specified path.
def save_wav(wav, path):
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	wavfile.write(path, config.sample_rate, wav.astype(np.int16))

# Pre-emphasis and inverse pre-emphasis
# These functions apply a pre-emphasis filter to the audio waveform to boost high frequencies and reduce low frequencies.
def preemphasis(x):
	return signal.lfilter([1, -config.preemphasis], [1], x)

# This function applies the inverse of the pre-emphasis filter to the audio waveform to restore the original signal.
# It takes the pre-emphasized waveform as input and returns the original waveform.
def inv_preemphasis(x):
	return signal.lfilter([1], [1, -config.preemphasis], x)

# Spectrogram and inverse spectrogram
# These functions convert the audio waveform to a spectrogram and vice versa.
def spectrogram(y):
	D = _stft(preemphasis(y))
	S = _amp_to_db(np.abs(D)) - config.ref_level_db
	return _normalize(S)

# This function converts the spectrogram back to a waveform using the Griffin-Lim algorithm.
# It takes the spectrogram as input and returns the reconstructed waveform.
def inv_spectrogram(spectrogram):
	"""
		Converts spectrogram to waveform using librosa
	"""
	S = _db_to_amp(_denormalize(spectrogram) + config.ref_level_db)  # Convert back to linear
	return inv_preemphasis(_griffin_lim(S ** config.power))          # Reconstruct phase

# This function converts the audio waveform to a mel-spectrogram.
# It takes the waveform as input and returns the mel-spectrogram.
def melspectrogram(y):
	D = _stft(preemphasis(y))
	S = _amp_to_db(_linear_to_mel(np.abs(D)))
	return _normalize(S)

# This function converts the mel-spectrogram back to a waveform using the Griffin-Lim algorithm.
# It takes the mel-spectrogram as input and returns the reconstructed waveform.
def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
	window_length = int(config.sample_rate * min_silence_sec)
	hop_length = int(window_length / 4)
	threshold = _db_to_amp(threshold_db)
	for x in range(hop_length, len(wav) - window_length, hop_length):
		if np.max(wav[x:x+window_length]) < threshold:
			return x + hop_length
	return len(wav)

# This function finds the endpoint of the audio waveform.
# It takes the waveform, threshold in decibels, and minimum silence duration as input.
def _griffin_lim(S):
	"""
		librosa implementation of Griffin-Lim
		Based on https://github.com/librosa/librosa/issues/434
	"""
	angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
	S_complex = np.abs(S).astype(np.complex)
	y = _istft(S_complex * angles)
	for i in range(config.griffin_lim_iters):
		angles = np.exp(1j * np.angle(_stft(y)))
		y = _istft(S_complex * angles)
	return y

# This function computes the Short-Time Fourier Transform (STFT) of the audio waveform.
# It takes the waveform as input and returns the STFT.
def _stft(y):
	n_fft, hop_length, win_length = _stft_parameters()
	return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

# This function computes the inverse Short-Time Fourier Transform (ISTFT) of the STFT.
# It takes the STFT as input and returns the reconstructed waveform.
def _istft(y):
	_, hop_length, win_length = _stft_parameters()
	return librosa.istft(y, hop_length=hop_length, win_length=win_length)

# This function computes the parameters for the STFT.
# It returns the number of FFT points, hop length, and window length.
def _stft_parameters():
	n_fft = (config.num_freq - 1) * 2
	hop_length = int(config.frame_shift_ms / 1000 * config.sample_rate)
	win_length = int(config.frame_length_ms / 1000 * config.sample_rate)
	return n_fft, hop_length, win_length


# CONVERSION FUNCTIONS #

_mel_basis = None

# This function converts the linear spectrogram to a mel-spectrogram.
# It takes the linear spectrogram as input and returns the mel-spectrogram.
def _linear_to_mel(spectrogram):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis()
	return np.dot(_mel_basis, spectrogram)

# This function builds the mel basis matrix for converting linear spectrogram to mel-spectrogram.
# It uses the librosa library to create the mel filter bank.
def _build_mel_basis():
	n_fft = (config.num_freq - 1) * 2
	return librosa.filters.mel(sr=config.sample_rate, n_fft=n_fft, n_mels=config.num_mels)

# This function converts the mel-spectrogram back to a linear spectrogram.
# It takes the mel-spectrogram as input and returns the linear spectrogram.
def _amp_to_db(x):
	return 20 * np.log10(np.maximum(1e-5, x))

# This function converts the decibel-scaled spectrogram back to amplitude.
# It takes the decibel-scaled spectrogram as input and returns the amplitude.
def _db_to_amp(x):
	return np.power(10.0, x * 0.05)

# This function normalizes the spectrogram to the range [0, 1].
# It takes the spectrogram as input and returns the normalized spectrogram.
def _normalize(S):
	return np.clip((S - config.min_level_db) / -config.min_level_db, 0, 1)

# This function denormalizes the spectrogram back to the original range.
# It takes the normalized spectrogram as input and returns the denormalized spectrogram.
def _denormalize(S):
	return (np.clip(S, 0, 1) * -config.min_level_db) + config.min_level_db
