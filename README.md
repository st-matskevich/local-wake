# local-wake

Lightweight wake word detection that runs locally and is suitable for resource-constrained devices like the Raspberry Pi. It requires no model training to support custom wake words and can be fully configured by end users on their devices. The system is based on feature extraction combined with time-warping comparison against a user-defined reference set.

## Installation
### Prerequisites
- Python 3.9 or later
- pip (Python package manager)
- Audio input device (e.g., microphone)

### Steps
```bash
# Clone the repository
git clone https://github.com/st-matskevich/local-wake.git
cd local-wake

# Create virtual environment
python -m venv .env
source .env/bin/activate

# Install required packages
pip install .
```

### System Dependencies
sounddevice package [depends on PortAudio which is not installed automatically on Linux](https://python-sounddevice.readthedocs.io/en/0.5.1/installation.html#installation). You can install it manually on Ubuntu with this:
```bash
sudo apt install libportaudio2
```

## Usage

### CLI Usage

#### Recording Reference Samples
The reference set is a collection of wake word recordings used as templates during detection. Usually, 3-4 samples are sufficient to achieve reliable detection performance.

```bash
lwake record ref/sample-1.wav
```
- `ref/sample-1.wav` - Path for the recorded file

Optional arguments:
- `--duration` (default: 3) - Duration in seconds
- `--no-vad` - Skip Voice Activity Detection silence trimming

Alternatively, you may use any recording tool of your choice. However, make sure that appropriate preprocessing is applied - specifically, silence must be trimmed from the recordings to achieve proper detection performance. A simple example of recording on Linux is:
```
arecord -d 3 -r 16000 -c 1 -f S16_LE output.wav
```

Notes:
 - Current Voice Activity Detection may occasionally be too aggressive, therefore, verify your recordings to ensure the wake word is fully captured and not inadvertently trimmed.

#### Audio Comparison
To evaluate comparison and determine a suitable detection threshold:

```bash
lwake compare ref/sample-1.wav ref/sample-2.wav
```
- `ref/sample-1.wav` - Path to the first file for comparison
- `ref/sample-2.wav` - Path to the second file for comparison

Optional arguments:
- `--method` (default: `embedding`) - Feature extraction method: `embedding` or `mfcc`

#### Real-time Detection
Once reference set is ready and threshold value has been identified:

```bash
lwake listen reference/folder 0.1 
```
- `reference/folder` - Directory containing your reference wake word .wav files
- `0.1` - Detection threshold. Adjust this value based on your comparison tests to balance sensitivity and false positives

Optional arguments:
- `--method` (default: `embedding`) - Feature extraction method: `embedding` or `mfcc`
- `--buffer-size` (default: 2.0) - Audio buffer size in seconds
- `--slide-size` (default: 0.25) - Step size in seconds for the sliding window
- `--debug` - Enable debug logs to observe real-time scores for incoming audio chunks

All logs are printed to stderr, while detection events are printed in JSON format to stdout:
```json
{"timestamp": 1754947173771, "wakeword": "sample-01.wav", "distance": 0.00943875619501332}
```

Notes:
- Buffer size should be similar to or slightly larger than your reference recording length
- Slide size can be set lower for better precision at the cost of higher CPU usage

### Library Usage

You can also use local-wake as a Python library:

```python
import lwake

# Record audio sample
lwake.record("sample.wav", duration=3, trim_silence=True)

# Compare two audio files
distance = lwake.compare("file1.wav", "file2.wav", method="embedding")
print(f"Distance: {distance}")

# Real-time detection (blocking call)
lwake.listen("reference/folder", threshold=0.1, method="embedding")
```

### Examples
This repository includes several pre-recorded examples for experimenting with the project. You can find them in the [examples](/examples) directory. While each example provides a suggested detection threshold, this value may require adjustment based on differences in microphone quality and environment.

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

## Evaluation

local-wake achieves **98.6% detection accuracy** on clean same-speaker audio using the [Qualcomm Keyword Speech Dataset](https://www.qualcomm.com/developer/software/keyword-speech-dataset).  

For detailed evaluation results, see the [benchmark documentation](benchmark/README.md).

## To do
- Consider using a small model on top of feature extraction for comparison instead of DTW
- Consider rebuilding the model to use a different inference engine
- Consider using VAD for audio preprocessing
- Consider using noise suppression for audio preprocessing
- Remove `tensorflow_hub` and load model from local files to allow completely offline usage

## Built with
- [Python](https://www.python.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [google/speech-embedding](https://www.kaggle.com/models/google/speech-embedding)
- [librosa](https://librosa.org/)
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [python-sounddevice](https://github.com/spatialaudio/python-sounddevice)

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## Contributing
Want a new feature added? Found a bug? 
Go ahead and open [a new issue](https://github.com/st-matskevich/local-wake/issues/new) or feel free to submit a pull request.