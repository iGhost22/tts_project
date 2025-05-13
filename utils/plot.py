import numpy as np
import librosa.display
from . import audio
from config import config
import matplotlib.pyplot as plt
plt.switch_backend('agg')


# CONSTANTS #
# These constants are used for plotting the spectrogram and alignment.
# They are defined in the config file and are used to set the sample rate, frame length, hop length, and FFT size.
# The sample rate is the number of samples per second, the frame length is the length of each frame in milliseconds,
fs = config.sample_rate
win = config.frame_length_ms
hop = config.frame_shift_ms
nfft = (config.num_freq - 1) * 2
hop_length = config.hop_length


# PLOT ALIGNMENT #
# This function plots the alignment between the encoder and decoder.
# It takes the alignment matrix, the path to save the plot, and an optional info string as arguments.
# The alignment matrix is a 2D array where each element represents the alignment score between the encoder and decoder.
# The path is the location where the plot will be saved.
# The info string is an optional argument that can be used to add additional information to the plot.
# The function uses the matplotlib library to create the plot and save it as a PNG file.
# The plot shows the alignment scores between the encoder and decoder over time.
# The x-axis represents the decoder time step, and the y-axis represents the encoder time step.
# The color of each cell in the plot represents the alignment score, with darker colors indicating higher scores.
# The function also adds a color bar to the plot to indicate the scale of the alignment scores.
# The plot is saved as a PNG file with a resolution of 300 DPI.
# The function also clears the current figure to avoid overlapping plots.
# It uses the imshow function from matplotlib to create the plot.
# The aspect parameter is set to 'auto' to allow the plot to adjust its aspect ratio automatically.
# The origin parameter is set to 'lower' to place the origin of the plot at the lower left corner.
# The interpolation parameter is set to 'none' to avoid any interpolation between the cells in the plot.
# The xlabel and ylabel parameters are used to set the labels for the x-axis and y-axis, respectively.
# The tight_layout function is used to adjust the layout of the plot to avoid overlapping labels and titles.
def plot_alignment(alignment, path, info=None):
	plt.gcf().clear()
	fig, ax = plt.subplots()
	im = ax.imshow(
		alignment,
		aspect='auto',
		origin='lower',
		interpolation='none')
	fig.colorbar(im, ax=ax)
	xlabel = 'Decoder timestep'
	if info is not None:
		xlabel += '\n\n' + info
	plt.xlabel(xlabel)
	plt.ylabel('Encoder timestep')
	plt.tight_layout()
	plt.savefig(path, dpi=300, format='png')
	plt.close()


# PLOT SPECTROGRAM #
# This function plots the spectrogram of the audio signal.
# It takes the linear output of the audio signal and the path to save the plot as arguments.
# The linear output is a 2D array where each element represents the amplitude of the audio signal at a specific time and frequency.
# The path is the location where the plot will be saved.
# The function uses the audio._denormalize function to denormalize the linear output before plotting.
def plot_spectrogram(linear_output, path):
	spectrogram = audio._denormalize(linear_output)
	plt.gcf().clear()
	plt.figure(figsize=(16, 10))
	plt.imshow(spectrogram.T, aspect="auto", origin="lower")
	plt.colorbar()
	plt.tight_layout()
	plt.savefig(path, dpi=300, format="png")
	plt.close()	


# TEST VISUALIZE #
# This function visualizes the alignment and spectrogram of the audio signal.
# It takes the alignment matrix, the spectrogram of the audio signal, and the path to save the plot as arguments.
# The alignment matrix is a 2D array where each element represents the alignment score between the encoder and decoder.
def test_visualize(alignment, spectrogram, path):
	
	_save_alignment(alignment, path)
	_save_spectrogram(spectrogram, path)
	label_fontsize = 16
	plt.gcf().clear()
	plt.figure(figsize=(16,16))
	
	plt.subplot(2,1,1)
	plt.imshow(alignment.T, aspect="auto", origin="lower", interpolation=None)
	plt.xlabel("Decoder timestamp", fontsize=label_fontsize)
	plt.ylabel("Encoder timestamp", fontsize=label_fontsize)
	plt.colorbar()

	plt.subplot(2,1,2)
	librosa.display.specshow(spectrogram.T, sr=fs, 
							 hop_length=hop_length, x_axis="time", y_axis="linear")
	plt.xlabel("Time", fontsize=label_fontsize)
	plt.ylabel("Hz", fontsize=label_fontsize)
	plt.tight_layout()
	plt.colorbar()

	plt.savefig(path + '_all.png', dpi=300, format='png')
	plt.close()


# SAVE ALIGNMENT #
# This function saves the alignment plot as a PNG file.
# It takes the alignment matrix and the path to save the plot as arguments.
# The alignment matrix is a 2D array where each element represents the alignment score between the encoder and decoder.
# The path is the location where the plot will be saved.
# The function uses the imshow function from matplotlib to create the plot.
def _save_alignment(alignment, path):
	plt.gcf().clear()
	plt.imshow(alignment.T, aspect="auto", origin="lower", interpolation=None)
	plt.xlabel("Decoder timestamp")
	plt.ylabel("Encoder timestamp")
	plt.colorbar()
	plt.savefig(path + '_alignment.png', dpi=300, format='png')


# SAVE SPECTROGRAM #
# This function saves the spectrogram plot as a PNG file.
# It takes the spectrogram of the audio signal and the path to save the plot as arguments.
# The spectrogram is a 2D array where each element represents the amplitude of the audio signal at a specific time and frequency.
# The path is the location where the plot will be saved.
# The function uses the pcolormesh function from matplotlib to create the plot.
# The pcolormesh function creates a pseudocolor plot with a non-regular rectangular grid.
def _save_spectrogram(spectrogram, path):
	plt.gcf().clear()  # Clear current previous figure
	cmap = plt.get_cmap('jet')
	t = win + np.arange(spectrogram.shape[0]) * hop
	f = np.arange(spectrogram.shape[1]) * fs / nfft
	plt.pcolormesh(t, f, spectrogram.T, cmap=cmap)
	plt.xlabel('Time (sec)')
	plt.ylabel('Frequency (Hz)')
	plt.colorbar()
	plt.savefig(path + '_spectrogram.png', dpi=300, format='png')
