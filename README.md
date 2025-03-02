# Capuchin Bird Call Detection

![Capuchin Bird](https://upload.wikimedia.org/wikipedia/commons/f/f7/Capuchin_bird.jpg)

## Overview
This project implements a deep learning model to detect Capuchin bird calls in audio recordings with **98% precision**. The model uses convolutional neural networks (CNNs) to analyze spectrograms generated from audio data and can automatically count the number of Capuchin bird calls in forest recordings.

## Features
- Audio preprocessing pipeline for WAV and MP3 files
- Spectrogram generation for audio analysis
- CNN-based classification model with 98% precision
- Batch processing capabilities for long recordings
- Tools for analyzing and visualizing results

## Model Architecture
The model uses a sequential architecture with:
- 2 Convolutional layers (8 and 16 filters)
- Flatten layer
- Dense layer with 64 neurons and ReLU activation
- Output layer with sigmoid activation for binary classification

## Dataset
The dataset consists of:
- Positive samples: Isolated Capuchin bird calls
- Negative samples: Other forest sounds and bird calls
- Full forest recordings for testing and deployment

## Results
- **Precision: 98%**
- High recall on test data
- Effective at isolating Capuchin calls in noisy forest environments

## Usage

### Prerequisitestensorflow>=2.4.0 tensorflow-io matplotlib numpy seaborn scikit-learnCopyCopy### Installation ```bash git clone https://github.com/yourusername/capuchin-bird-detection.git cd capuchin-bird-detection pip install -r requirements.txt
Running Detection on Forest RecordingsCopypythonCopy
import tensorflow as tf from detection import load_mp3_16k_mono, preprocess_mp3 # Load the trained model model = tf.keras.models.load_model('capuchin_model.h5') # Process a forest recording wav = load_mp3_16k_mono('path_to_recording.mp3') audio_slices = tf.keras.utils.timeseries_dataset_from_array( wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1 ) audio_slices = audio_slices.map(preprocess_mp3) audio_slices = audio_slices.batch(64) # Get predictions predictions = model.predict(audio_slices)
Visualization
The repository includes scripts for generating:
* Waveform visualizations
* Spectrogram representations
* Confusion matrices
Future Work
* Apply transfer learning to detect other rainforest species
* Implement real-time detection on edge devices
* Expand the dataset with more diverse environmental conditions
Citation
If you use this code for your research, please cite:CopyCopy@software{capuchin_detection, author = {Your Name}, title = {Capuchin Bird Call Detection}, year = {2025}, url = {https://github.com/yourusername/capuchin-bird-detection} }
License
This project is licensed under the MIT License - see the LICENSE file for details.Copyconvert all into markdownEditI notice there are some formatting issues in your markdown. Let me correct the formatting to create proper markdown:
markdownCopy# Capuchin Bird Call Detection

![Capuchin Bird](https://upload.wikimedia.org/wikipedia/commons/f/f7/Capuchin_bird.jpg)

## Overview
This project implements a deep learning model to detect Capuchin bird calls in audio recordings with **98% precision**. The model uses convolutional neural networks (CNNs) to analyze spectrograms generated from audio data and can automatically count the number of Capuchin bird calls in forest recordings.

## Features
- Audio preprocessing pipeline for WAV and MP3 files
- Spectrogram generation for audio analysis
- CNN-based classification model with 98% precision
- Batch processing capabilities for long recordings
- Tools for analyzing and visualizing results

## Model Architecture
The model uses a sequential architecture with:
- 2 Convolutional layers (8 and 16 filters)
- Flatten layer
- Dense layer with 64 neurons and ReLU activation
- Output layer with sigmoid activation for binary classification

## Dataset
The dataset consists of:
- Positive samples: Isolated Capuchin bird calls
- Negative samples: Other forest sounds and bird calls
- Full forest recordings for testing and deployment

## Results
- **Precision: 98%**
- High recall on test data
- Effective at isolating Capuchin calls in noisy forest environments

## Usage

### Prerequisites
tensorflow>=2.4.0
tensorflow-io
matplotlib
numpy
seaborn
scikit-learn
Copy
### Installation
git clone https://github.com/yourusername/capuchin-bird-detection.git
cd capuchin-bird-detection
pip install -r requirements.txt
Running Detection on Forest Recordings
pythonCopyimport tensorflow as tf
from detection import load_mp3_16k_mono, preprocess_mp3

# Load the trained model
model = tf.keras.models.load_model('capuchin_model.h5')

# Process a forest recording
wav = load_mp3_16k_mono('path_to_recording.mp3')
audio_slices = tf.keras.utils.timeseries_dataset_from_array(
    wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1
)
audio_slices = audio_slices.map(preprocess_mp3)
audio_slices = audio_slices.batch(64)

# Get predictions
predictions = model.predict(audio_slices)

# Visualization
The repository includes scripts for generating:

Waveform visualizations
Spectrogram representations
Confusion matrices

## Future Work

Apply transfer learning to detect other rainforest species
Implement real-time detection on edge devices
Expand the dataset with more diverse environmental conditions
