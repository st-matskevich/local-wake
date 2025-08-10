import sounddevice as sd
import soundfile as sf
import argparse
import os

def record_audio(filename, duration=3, sample_rate=16000):
    print(f"Recording for {duration} seconds...")

    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    sf.write(filename, audio, samplerate=sample_rate)
    print(f"Saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record audio and save as WAV.")
    parser.add_argument("--out", type=str, required=True, help="Output .wav file path")
    parser.add_argument("--duration", type=int, default=3, help="Duration in seconds")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate in Hz")

    args = parser.parse_args()
    record_audio(args.out, args.duration, args.sr)
