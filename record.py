import sounddevice as sd
import soundfile as sf
import argparse
import logging
import os

_logger = logging.getLogger("local-wake")

def record_audio(filename, duration=3, sample_rate=16000):
    _logger.info(f"Recording for {duration} seconds...")

    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    sf.write(filename, audio, samplerate=sample_rate)
    _logger.info(f"Saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Record audio and save as WAV.")
    parser.add_argument("out", type=str, help="Output .wav file path")
    parser.add_argument("--duration", type=int, default=3, help="Duration in seconds")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate in Hz")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    record_audio(args.out, args.duration, args.sr)

if __name__ == "__main__":
    main()
