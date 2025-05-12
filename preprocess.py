import os
import glob
import librosa
import argparse
from utils import data
from tqdm import tqdm
from config import config, get_preprocess_args


# MAKE META #
# This function creates a metadata file from the text and audio files.
# It uses the data.build_from_path function to create a list of metadata entries, 
# and then writes this list to a file using data.write_meta_data.
# The metadata file contains information about the audio files, such as their duration and sample rate.
# The function takes the following arguments:
# - text_input_path: The path to the text file containing the metadata.
# - audio_input_dir: The directory containing the audio files.
# - meta_dir: The directory where the metadata file will be saved.
# - meta_text: The name of the metadata file.
# - file_suffix: The suffix of the audio files (e.g., 'wav').
# - num_workers: The number of workers to use for processing the audio files.
# - frame_shift_ms: The frame shift in milliseconds.
def make_meta(text_input_path, audio_input_dir, meta_dir, meta_text, file_suffix, num_workers, frame_shift_ms):
	os.makedirs(meta_dir, exist_ok=True)
	metadata = data.build_from_path(text_input_path, audio_input_dir, meta_dir, file_suffix, num_workers, tqdm=tqdm)
	data.write_meta_data(metadata, meta_dir, meta_text, frame_shift_ms)


# DATASET ANALYSIS #
# This function analyzes the dataset by loading the audio files and calculating their duration.
# It uses the librosa library to load the audio files and calculate their duration.
# The function takes the following arguments:
# - wav_dir: The directory containing the audio files.
# - file_suffix: The suffix of the audio files (e.g., 'wav').

def dataset_analysis(wav_dir, file_suffix):

	audios = sorted(glob.glob(os.path.join(wav_dir, '*.' + file_suffix)))
	print('Training data count: ', len(audios))

	duration = 0.0
	max_d = 0
	min_d = 60
	for audio in tqdm(audios):
		y, sr = librosa.load(audio)
		d = librosa.get_duration(y=y, sr=sr)
		if d > max_d: max_d = d
		if d < min_d: min_d = d
		duration += d

	print('Sample rate: ', sr)
	print('Speech total length (hr): ', duration / 60**2)
	print('Max duration (seconds): ', max_d)
	print('Min duration (seconds): ', min_d)
	print('Average duration (seconds): ', duration / len(audios))


# MAIN #
def main():

	args = get_preprocess_args()

	if args.mode == 'all' or args.mode == 'make' or args.mode == 'analyze':
		
		#---preprocess text and data to be model ready---#
		if args.mode == 'all' or args.mode == 'make':
			make_meta(args.text_input_path, args.audio_input_dir, args.meta_dir, args.meta_text, args.file_suffix, args.num_workers, config.frame_shift_ms)

		#---dataset analyze---#
		if args.mode == 'all' or args.mode == 'analyze':
			dataset_analysis(args.audio_input_dir, args.file_suffix)
	
	else:
		raise RuntimeError('Invalid mode!')



if __name__ == '__main__':
	main()