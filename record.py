import sounddevice as sd
import soundfile as sf
import argparse
import logging
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

_logger = logging.getLogger("local-wake")

def trim_silence_with_vad(audio, sample_rate):
    _logger.info("Loading Silero VAD model...")
    model = load_silero_vad()

    t = torch.from_numpy(audio[:, 0]).float()
    speech_timestamps = get_speech_timestamps(
        t, model,
        sampling_rate=sample_rate,
    )
    
    if not speech_timestamps:
        _logger.warning("No speech detected in audio")
        return audio
    
    start_sample = speech_timestamps[0]['start']
    end_sample = speech_timestamps[-1]['end']
    
    _logger.info(f"Trimmed audio to [{start_sample/sample_rate:.2f}s, {end_sample/sample_rate:.2f}s]")
    return audio[start_sample:end_sample]

def record_audio(filename, duration=3, sample_rate=16000, trim_silence=True):
    _logger.info(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    if trim_silence:
        _logger.info("Trimming silence using VAD...")
        audio = trim_silence_with_vad(audio, sample_rate)
        
    sf.write(filename, audio, samplerate=sample_rate)
    _logger.info(f"Saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Record audio and save as WAV.")
    parser.add_argument("out", type=str, help="Output .wav file path")
    parser.add_argument("--duration", type=int, default=3, help="Duration in seconds")
    parser.add_argument("--sr", type=int, default=16000, help="Sample rate in Hz")
    parser.add_argument("--no-vad", action="store_true", help="Skip VAD silence trimming")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    record_audio(args.out, args.duration, args.sr, trim_silence=not args.no_vad)

if __name__ == "__main__":
    main()