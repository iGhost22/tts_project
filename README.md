# Text-to-Speech (TTS) Project - Tacotron

A Text-to-Speech project using Tacotron model to convert text into natural speech.

## 📋 Table of Contents

- [Demo](#-demo)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Model Training](#-model-training)
- [Using the Model](#-using-the-model)
- [API Server](#-api-server)
- [Model Evaluation](#-model-evaluation)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)

## 🎵 Demo

Explore our Text-to-Speech project through comprehensive demonstrations and documentation:

### 📋 Project Report

Access the complete thesis report and technical documentation:

- **Thesis Report**: [📄 Full research documentation](https://drive.google.com/file/d/1WL36MKCzDwwJ7Wselvcy_ll7LsBPz4-N/view)

### 🎬 Google Drive Demonstrations

Watch detailed demonstrations of the TTS system in action:

- **Demo Videos**: [📁 Google Drive demonstrations folder](https://drive.google.com/drive/folders/1WduADe8MMtqWW-3eXy1phVqTwHNFMU62)

## 🔧 System Requirements

### Hardware

- **GPU**: NVIDIA GPU with CUDA support (RTX 3060 or better recommended)
- **RAM**: Minimum 8GB, 16GB or more recommended
- **Storage**: Minimum 10GB free space

### Software

- **Python**: 3.7 - 3.9
- **CUDA**: 10.2 or 11.x (compatible with PyTorch)
- **Git**: For cloning repository

## 🚀 Installation

### 1. Clone repository

```bash
git clone https://github.com/iGhost22/tts_project.git
cd tts_project
```

### 2. Create virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install NLTK data (required for text preprocessing)

```python
import nltk
nltk.download('punkt')
```

## 📊 Data Preparation

### 1. Data structure

Create data directory with the following structure:

```
data/
├── wavs/           # Audio files (.wav)
├── metadata.csv    # Metadata file
└── meta/          # Directory for processed data
```

### 2. Metadata format

The `metadata.csv` file should have the format:

```
filename|transcript
audio1.wav|This is the first text sentence
audio2.wav|This is the second text sentence
```

### 3. Data preprocessing

```bash
python preprocess.py --mode all --meta_dir ./data/meta --meta_text meta_text.txt
```

**Parameters:**

- `--mode`: Preprocessing mode (`make`, `analyze`, `all`)
- `--meta_dir`: Directory to save processed data
- `--meta_text`: Output metadata filename

## 🏋️ Model Training

### 1. Basic training

```bash
python train.py --ckpt_dir ckpt/ --log_dir log/
```

### 2. Resume training from checkpoint

```bash
python train.py --ckpt_dir ckpt/ --log_dir log/ --model_name 500000
```

**Training parameters:**

- `--data_root`: Path to preprocessed data
- `--meta_text`: Transcript filename
- `--ckpt_dir`: Checkpoint directory
- `--model_name`: Checkpoint name to resume training
- `--log_dir`: Log directory for TensorBoard
- `--log_comment`: Comment for logs

## 🎯 Using the Model

### 1. Synthesize speech from text

```bash
python test.py --interactive --plot --model_name 500000
```

### 2. Batch synthesis

```bash
python3 test.py --plot --model_name 500000 --test_file_path ./data/test_transcripts.txt
```

**Parameters:**

- `--text`: Text to synthesize
- `--text_file`: File containing list of texts
- `--model_name`: Checkpoint to use
- `--result_dir`: Output directory

## 🌐 API Server

### 1. Start API server

```bash
cd tts_api
python app.py
```

### 2. Using the API

**Endpoint:** `POST /generate-speech`

```bash
curl -X POST "http://localhost:8000/generate-speech" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, this is a speech synthesis API"}'
```

**Response:** WAV audio file

### 3. API Documentation

Access `http://localhost:8000/docs` to view Swagger documentation.

## 📈 Model Evaluation

### 1. Evaluation with automatic metrics

```bash
python evaluate.py
```

**Evaluated metrics:**

- **MCD** (Mel-Cepstral Distortion): Measures mel spectrum distortion
- **LogF0RMSE**: Measures fundamental frequency error
- **UTMOS**: Overall quality assessment
- **SpeechBERTScore**: Semantic similarity evaluation

## 📁 Project Structure

```
tts_project/
├── config.py                 # Model configuration
├── dataloader.py             # Data loader
├── train.py                  # Training script
├── test.py                   # Inference script
├── evaluate.py               # Evaluation script
├── preprocess.py             # Preprocessing script
├── requirements.txt          # Dependencies
├── model/                    # Tacotron model
│   ├── tacotron.py
│   └── loss.py
├── utils/                    # Utilities
│   ├── audio.py
│   ├── text.py
│   └── plot.py
├── tts_api/                  # API server
│   ├── app.py
│   ├── inference.py
│   └── requirements.txt
├── data/                     # Data
├── ckpt/                     # Checkpoints
├── log/                      # TensorBoard logs
└── result/                   # Output results
```

## ⚙️ Configuration

### Audio configuration

```python
num_mels = 80              # Number of mel bands
num_freq = 1025            # Number of frequency bins
sample_rate = 22050        # Sample rate
frame_length_ms = 50       # Frame length (ms)
frame_shift_ms = 12.5      # Frame shift (ms)
```

### Model configuration

```python
embedding_dim = 256        # Embedding dimension
outputs_per_step = 5       # Mel frames per decoder step
attention = 'LocationSensitive'  # Attention type
```

### Training configuration

```python
batch_size = 8             # Batch size
initial_learning_rate = 0.002  # Initial learning rate
max_epochs = 1000          # Maximum epochs
max_steps = 500000         # Maximum steps
checkpoint_interval = 2000 # Checkpoint save interval
```

## 🔧 Troubleshooting

### CUDA errors

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Memory errors

- Reduce `batch_size` in `config.py`
- Reduce `num_workers` in dataloader

### Dependency errors

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## 📝 License

This project is distributed under the MIT License. See `LICENSE` file for more details.

## 🤝 Contributing

All contributions are welcome! Please create an issue or pull request.
