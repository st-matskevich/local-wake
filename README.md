# local-wake

Lightweight wake word detection that runs locally and is suitable for resource-constrained devices like the Raspberry Pi. It requires no model training to support custom wake words and can be fully configured by end users on their devices. The system is based on feature extraction combined with time-warping comparison against a user-defined reference set.

## Installation
### Prerequisites
- Python 3.8 or later
- pip (Python package manager)
- Audio input device (e.g., microphone)

### Steps
```
# Clone the repository
git clone https://github.com/st-matskevich/local-wake.git
cd local-wake

# Create virtual environment
python -m venv .env
source .env/bin/activate

# Install required packages
pip install tensorflow tensorflow-hub librosa sounddevice soundfile numpy

# Install PortAudio system library for sounddevice if it wasn't installed via pip
sudo apt install libportaudio2
```

## Usage
### Reference set
The reference set is a collection of wake word recordings used as templates during detection. Usually, 3-4 samples are sufficient to achieve reliable detection performance.

This repository includes `record.py` to record samples:
```
python record.py ref/sample-1.wav --duration 3
```

Alternatively, you may use any recording tool of your choice. For example, on Linux:
```
arecord -d 3 -r 16000 -c 1 -f S16_LE output.wav
```

Notes:
 - The recording script is quite straightforward and may capture silence or background noise before/after the wake word during default 3-second capture. For better detection precision, it's recommended to trim recordings down to just the wake word segment. This can be done manually using audio editing software or automatically using Voice Activity Detection (VAD) tools.

### Manual audio comparison
To evaluate comparison and determine a suitable detection threshold you can use `compare.py`:
```
python compare.py ref/sample-1.wav ref/sample-2.wav
```
- ref/sample-1.wav - Path to the first file for comparison.
- ref/sample-2.wav - Path to the second file for comparison.

Optional arguments:
- `--method` (default: `embedding`) - Feature extraction method, `embedding` or `mfcc`.

### Real-time detection
Once reference set is ready and threshold value has been identified, you can use `listen.py` to start real-time detection:
```
python listen.py reference/folder 0.1 
```
- reference/folder - Directory containing your reference wake word .wav files.
- 0.1 - Detection threshold. Adjust this value based on your comparison tests to balance sensitivity and false positives.

Optional arguments:
- `--method` (default: `embedding`) - Feature extraction method, `embedding` or `mfcc`.
- `--buffer-size` (default: 2.0) - Audio buffer size in seconds.
- `--slide-size` (default: 0.25) - Step size in seconds for the sliding window.
- `--debug` - Enable capture debug logs

All logs are printed to stderr, while detection events are printed in JSON format to stdout to simplify parsing:
`{"timestamp": 1754947173771, "wakeword": "sample-01.wav", "distance": 0.00943875619501332}`

Notes:
- Buffer size should be similar to or slightly larger than your reference recording length
- Slide size can be set to a lower value for better precision at the cost of higher CPU usage

## Implementation
Existing solutions for wake word detection can generally be divided into two categories:
- Classical deterministic, speaker-dependent approaches - Typically based on MFCC feature extraction combined with DTW, as used in projects such as [Rhasspy Raven](https://github.com/rhasspy/rhasspy-wake-raven) or [Snips](https://medium.com/snips-ai/machine-learning-on-voice-a-gentle-introduction-with-snips-personal-wake-word-detector-133bd6fb568e).
  - Advantages: Support for user-defined wake words with minimal development effort.
  - Limitations: Strongly speaker-dependent, requiring sample collection from all intended users. Highly sensitive to background noise.

- Modern model-based, speaker-independent approaches - Use neural models to classify wake words directly, as in [openWakeWord](https://github.com/dscripka/openWakeWord) or [Porcupine](https://github.com/Picovoice/porcupine).
  - Advantages: High precision across multiple speakers without additional sample collection.
  - Limitations: Do not support arbitrary user-defined wake words. Adapting to product-specific wake words requires model retraining or fine-tuning, which, depending on the solution, can be complex and typically requires at least a basic understanding of machine learning concepts and dataset preparation.

Choosing either category imposes strict limitations: deterministic methods sacrifice robustness, while model-based methods sacrifice adaptability.

local-wake combines neural feature extraction with classical sequence matching to achieve flexible and robust wake word detection. It uses a pretrained Google's [speech-embedding](https://www.kaggle.com/models/google/speech-embedding) model to extract speech features, then applies Dynamic Time Warping to compare incoming audio against a user-defined reference set of wake word samples.

This approach merges the advantages of both categories described above: it supports user-defined wake words like traditional deterministic methods, while benefiting from the enhanced feature representations and noise robustness provided by neural models. The result is a system that delivers good precision and flexibility without requiring extensive model training or large datasets.

## To do
- Consider using fast DTW implementation to reduce CPU usage
- Consider using a small model on top of feature extraction for comparison instead of DTW
- Consider using VAD for audio preprocessing
- Consider using noise suppression for audio preprocessing
- Perform accuracy testing
- Remove `tensorflow_hub` and load model from local files to allow completely offline usage

## Built with
- [Python](https://www.python.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [google/speech-embedding](https://www.kaggle.com/models/google/speech-embedding)
- [librosa](https://librosa.org/)
- [python-sounddevice](https://github.com/spatialaudio/python-sounddevice)

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## Contributing
Want a new feature added? Found a bug? 
Go ahead and open [a new issue](https://github.com/st-matskevich/local-wake/issues/new) or feel free to submit a pull request.