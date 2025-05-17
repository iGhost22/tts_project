import os
import sys
import torch
from torch.autograd import Variable
import time

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import numpy with error handling
try:
    import numpy as np
except ImportError:
    print("ERROR: NumPy không thể import. Đảm bảo NumPy được cài đặt đúng cách.")
    raise

# Try to import other dependencies with error handling
try:
    import soundfile as sf
except ImportError:
    print("ERROR: SoundFile không thể import. Đảm bảo SoundFile được cài đặt đúng cách.")
    raise

# Now import from project modules
try:
    from utils import audio
    from utils.text import text_to_sequence, symbols
    from utils.plot import test_visualize
    from model.tacotron import Tacotron
    from config import config
except ImportError as e:
    print(f"ERROR: Không thể import module dự án: {str(e)}")
    raise

# Kiểm tra xem CUDA có sẵn không và có nên sử dụng không
USE_CUDA = torch.cuda.is_available() and os.environ.get("USE_CUDA", "0") == "1"
print(f"CUDA available: {torch.cuda.is_available()}, USE_CUDA set to: {USE_CUDA}")

# Check numpy version
print(f"NumPy version: {np.__version__}")

# Validate numpy is working properly by creating a small array
try:
    test_array = np.array([1, 2, 3])
    print(f"NumPy test successful: {test_array}")
except Exception as e:
    print(f"ERROR: NumPy test failed: {str(e)}")
    raise

class TTSModel:
    def __init__(self, checkpoint_path=None):
        self.model = None
        # Default checkpoint path, can be overridden by environment variable
        self.checkpoint_path = checkpoint_path or os.environ.get("CHECKPOINT_PATH", "../ckpt/checkpoint_step500000.pth")
        self.result_dir = "result"
        self.output_file = "output.wav"
        self.figure_file = "alignment_spec"  # Tên cố định cho file hình ảnh
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
        # Tăng max_decoder_steps lên gấp đôi để xử lý văn bản dài tốt hơn
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
        print(f"TTS method called with text: '{text}'")
        
        # Kiểm tra văn bản đầu vào
        if not text or not text.strip():
            raise ValueError("Văn bản đầu vào không được để trống")

        try:
            print("Converting text to sequence...")
            sequence = np.array(text_to_sequence(text))
            print(f"Sequence shape: {sequence.shape}, data: {sequence[:10]}{'...' if len(sequence) > 10 else ''}")
            
            print("Converting sequence to tensor...")
            sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
            print(f"Tensor shape: {sequence.shape}")

            if USE_CUDA:
                print("Moving tensor to CUDA...")
                sequence = sequence.cuda()

            # Greedy decoding với attention cao hơn
            print("Running model inference...")
            try:
                mel_outputs, linear_outputs, gate_outputs, alignments = self.model(sequence)
                print(f"Model inference successful, outputs shapes: mel={mel_outputs.shape}, linear={linear_outputs.shape}, gate={gate_outputs.shape}, alignments={alignments.shape}")
            except Exception as model_error:
                print(f"ERROR during model inference: {str(model_error)}")
                import traceback
                print(traceback.format_exc())
                raise RuntimeError(f"Model inference failed: {str(model_error)}")
                
            print("Converting outputs to numpy arrays...")
            try:
                linear_output = linear_outputs[0].cpu().data.numpy()
                print(f"Linear output shape: {linear_output.shape}")
                
                spectrogram = audio._denormalize(linear_output)
                print(f"Spectrogram shape: {spectrogram.shape}")
                
                alignment = alignments[0].cpu().data.numpy()
                print(f"Alignment shape: {alignment.shape}")
            except Exception as convert_error:
                print(f"ERROR converting outputs to numpy: {str(convert_error)}")
                import traceback
                print(traceback.format_exc())
                raise RuntimeError(f"Failed to convert model outputs: {str(convert_error)}")

            # Chuyển spectrogram thành waveform với chất lượng cao
            print("Converting spectrogram to waveform...")
            try:
                waveform = audio.inv_spectrogram(linear_output.T)
                print(f"Waveform generated, shape: {waveform.shape}, dtype: {waveform.dtype}")
                print(f"Waveform stats - min: {np.min(waveform)}, max: {np.max(waveform)}, mean: {np.mean(waveform)}")
            except Exception as wav_error:
                print(f"ERROR converting spectrogram to waveform: {str(wav_error)}")
                import traceback
                print(traceback.format_exc())
                raise RuntimeError(f"Failed to convert spectrogram to waveform: {str(wav_error)}")

            return waveform, alignment, spectrogram
            
        except Exception as e:
            print(f"ERROR in tts method: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise

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
        # Always use the fixed output file
        output_path = os.path.join(self.result_dir, self.output_file)
        output_without_ext = (
            output_path.rsplit(".", 1)[0] if "." in output_path else output_path
        )
        print(f"Output path: {output_path}, output without ext: {output_without_ext}")
        print(f"Result dir exists: {os.path.exists(self.result_dir)}, is dir: {os.path.isdir(self.result_dir) if os.path.exists(self.result_dir) else 'N/A'}")

        print(f"Synthesizing text: '{text}'")
        try:
            print("Calling tts method...")
            waveform, alignment, spectrogram = self.tts(text)
            print(f"tts completed, waveform shape: {waveform.shape}, alignment shape: {alignment.shape}, spectrogram shape: {spectrogram.shape}")

            # Tạo hình ảnh visualization nếu yêu cầu
            if generate_figures:
                # Sử dụng tên file cố định cho hình ảnh thay vì dùng timestamp
                figure_path = os.path.join(self.result_dir, self.figure_file)
                print(f"Generating visualization to: {figure_path}")
                try:
                    test_visualize(alignment, spectrogram, figure_path)
                    print(f"Đã lưu hình ảnh: {figure_path}.png")
                except Exception as vis_error:
                    print(f"WARNING: Visualization failed: {str(vis_error)}")
                    import traceback
                    print(traceback.format_exc())

            # Cải thiện chất lượng âm thanh
            if enhance_audio:
                print("Enhancing audio...")
                try:
                    max_abs = np.max(np.abs(waveform))
                    print(f"Max abs value: {max_abs}")
                    waveform = waveform * (0.95 / max(0.01, max_abs))
                    print("Audio enhancement completed")
                except Exception as enhance_error:
                    print(f"WARNING: Audio enhancement failed: {str(enhance_error)}")
                    import traceback
                    print(traceback.format_exc())

            # Lưu file với định dạng chất lượng cao (24-bit thay vì 16-bit)
            print(f"Writing audio file to: {output_without_ext}.wav with sample rate: {config.sample_rate}")
            try:
                sf.write(
                    output_without_ext + ".wav",
                    waveform,
                    config.sample_rate,
                    subtype="PCM_24",
                )
                print(f"Đã lưu âm thanh: {output_without_ext}.wav")
            except Exception as write_error:
                print(f"ERROR: Failed to write audio file: {str(write_error)}")
                import traceback
                print(traceback.format_exc())
                raise

            return output_without_ext + ".wav"

        except Exception as e:
            print(f"ERROR in synthesize_speech: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise


# Initialize model with default settings
# This can be imported and used in FastAPI app
tts_model = TTSModel()


# Function to be called from FastAPI
def generate_speech(text, filename=None):
    """
    Generate speech from text and return the path to the audio file

    Args:
        text: Text to synthesize
        filename: Ignored, always uses fixed output filename

    Returns:
        Path to the generated audio file
    """
    try:
        print(f"generate_speech called with text: '{text}'")
        
        # Check if numpy is available
        print(f"NumPy is available: {np is not None}, version: {np.__version__}")
        
        # Generate a more informative error if NumPy array creation fails
        try:
            # Test numpy functionality again right before use
            test_arr = np.array([1, 2, 3])
            print(f"NumPy array test passed within generate_speech: {test_arr}")
        except Exception as e:
            print(f"NumPy array creation failed within generate_speech: {str(e)}")
            raise ImportError(f"NumPy is available but not working properly: {str(e)}")
        
        # Create directory if not exists
        try:
            result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
            os.makedirs(result_dir, exist_ok=True)
            print(f"Result directory checked/created: {result_dir}")
        except Exception as e:
            print(f"Error creating result directory: {str(e)}")
            raise
            
        # Call TTS model
        print("Calling synthesize_speech...")
        audio_path = tts_model.synthesize_speech(text, generate_figures=True, enhance_audio=True)
        print(f"synthesize_speech completed, audio path: {audio_path}")
        
        # Verify file exists
        if os.path.exists(audio_path):
            file_size = os.path.getsize(audio_path)
            print(f"Audio file exists, size: {file_size} bytes")
        else:
            print(f"WARNING: Audio file does not exist at {audio_path}")
            
        return audio_path
    except Exception as e:
        error_message = f"Error generating speech: {str(e)}"
        print(error_message)
        # Log more details about the error
        import traceback
        print(f"Detailed traceback: {traceback.format_exc()}")
        raise Exception(error_message)
