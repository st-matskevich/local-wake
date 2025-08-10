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
The reference set is a collection of wake word recordings used as templates during detection. Typically, 3–4 samples are sufficient to achieve reliable detection performance.

This repository includes `record.py` to record samples:
```
python record.py --out ref/sample-1.wav --duration 3
```

Alternatively, you may use any recording tool of your choice. For example, on Linux, you can use arecord:
```
arecord -d 3 -r 16000 -c 1 -f S16_LE output.wav
```

### Manual audio comparison
To evaluate comparison and determine a suitable detection threshold you can use `compare.py`:
```
python compare.py ref/sample-1.wav ref/sample-2.wav
```

The script includes a `--method` option to switch between embedding-based and MFCC feature extraction. It is recommended to use this option only if you are familiar with the differences between these methods and wish to compare performance or precision with the deterministic algorithm.

### Real-time detection
Once reference set is ready and threshold value is identified, you can use `listen.py` to start real-time detection:
```
python listen.py reference/folder 0.1 
```


## Implementation
Define problem and existing solutions. Lack of user-side training with arbitary words.
Define implementation

## To do
- Consider using fast DTW implementation to reduce CPU consumption even more
- Consider using a small model on top of feature extraction for comparison instead of DTW
- Consider using VAD for audio preprocessing
- Consider using noice suppression for audio preprocessing
- Perform accuracy testing
- Update scripts output to make it easily parsable
- Remove `tensorflow_hub` and load model from local files to avoid internet access

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