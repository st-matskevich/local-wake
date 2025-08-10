# local-wake

Python scripts to recognize user-defined wake words in real time capable running on edge devices (tested on Raspberry Pi 4). Based on feature extraction and time-warping comparison with user-defined support set. 

# Installation
```
# Clone the repository
git clone https://github.com/st-matskevich/local-wake.git
cd local-wake

# Create virtual environment
python -m venv .env
source .env/bin/activate

# Install required packages
pip install tensorflow tensorflow-hub librosa sounddevice numpy
```

# Usage
Define how to use scripts

# Preabmle
Define problem and existing solutions. Lack of user-side training with arbitary words.

# Implementation
Define implementation

# To do
- Consider using fast DTW implementation to reduce CPU consumption even more
- Consider using a small model on top of feature extraction for comparison instead of DTW
- Consider using VAD for audio preprocessing
- Consider using noice suppression for audio preprocessing
- Perform accuracy testing
- Update scripts output to make it easily parsable

# Built with
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