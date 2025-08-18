import sounddevice as sd
import soundfile as sf
import logging
from silero_vad import load_silero_vad, get_speech_timestamps

_logger = logging.getLogger("local-wake")

def trim_silence_with_vad(audio, sample_rate):
    _logger.info("Loading Silero VAD model...")
    model = load_silero_vad()

    speech_timestamps = get_speech_timestamps(
        audio[:, 0], model,
        sampling_rate=sample_rate,
    )
    
    if not speech_timestamps:
        _logger.warning("No speech detected in audio")
        return audio
    
    start_sample = speech_timestamps[0]['start']
    end_sample = speech_timestamps[-1]['end']
    
    _logger.info(f"Trimmed audio to [{start_sample/sample_rate:.2f}s, {end_sample/sample_rate:.2f}s]")
    return audio[start_sample:end_sample]

def record(output, duration=3, trim_silence=True):
    """Record audio and save as WAV file"""
    _logger.info(f"Recording for {duration} seconds...")
    sample_rate=16000
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()

    if trim_silence:
        _logger.info("Trimming silence using VAD...")
        audio = trim_silence_with_vad(audio, sample_rate)
        
    sf.write(output, audio, samplerate=sample_rate)
    _logger.info(f"Saved to {output}")