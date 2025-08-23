import os
import sys
import time
import json
import numpy as np
import sounddevice as sd
import logging

_logger = logging.getLogger("local-wake")

def load_support_set(support_folder, method="embedding"):
    """Load reference wake word files from support folder"""
    from .features import extract_mfcc_features, extract_embedding_features
    
    support = []
    
    for file in os.listdir(support_folder):
        if not file.endswith(".wav"):
            continue
        
        path = os.path.join(support_folder, file)
        _logger.info(f"Loading reference file: {file}")
        
        try:
            if method == "mfcc":
                features = extract_mfcc_features(path=path)
            else:  # embedding
                features = extract_embedding_features(path=path)
            
            if features is not None:
                support.append((file, features))
                _logger.info(f"Features shape: {features.shape}")
        except Exception as e:
            _logger.error(f"Error loading {file}: {e}")
    
    return support

def listen(support_folder, threshold, method="embedding", buffer_size=2.0, slide_size=0.25, callback=None):
    """Real-time wake word detection"""
    from .features import extract_mfcc_features, extract_embedding_features, dtw_cosine_normalized_distance

    if callback is None:
        def callback(detection, _):
            print(json.dumps(detection), file=sys.stdout, flush=True)
    
    _logger.info(f"Loading support set using {method} features...")
    support_set = load_support_set(support_folder, method=method)
    
    if not support_set:
        _logger.error("No valid wake word files found in support folder")
        return
    
    _logger.info(f"Loaded {len(support_set)} reference files")
    
    sample_rate = 16000
    buffer_size_samples = int(buffer_size * sample_rate)
    slide_size_samples = int(slide_size * sample_rate)
    audio_buffer = np.zeros(buffer_size_samples, dtype=np.float32)
    
    _logger.info(f"Starting audio stream (buffer: {buffer_size}s, slide: {slide_size}s)")
    _logger.info(f"Using {method} features with threshold {threshold}")
    _logger.info("Listening for wake words...")
    
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.float32) as stream:
        while True:
            data, overflowed = stream.read(slide_size_samples)
            if overflowed:
                _logger.warning("Audio buffer overflowed")
            
            chunk = data[:, 0]
            audio_buffer = np.roll(audio_buffer, -len(chunk))
            audio_buffer[-len(chunk):] = chunk
            
            try:
                if method == "mfcc":
                    features = extract_mfcc_features(y=audio_buffer, sample_rate=sample_rate)
                else:  # embedding
                    features = extract_embedding_features(y=audio_buffer, sample_rate=sample_rate)
                
                if features is None:
                    continue
                
            except Exception as e:
                _logger.error(f"Feature extraction failed: {e}")
                continue
            
            timestamp = int(time.time() * 1000)
            for filename, ref_features in support_set:
                try:
                    distance = dtw_cosine_normalized_distance(features, ref_features)
                    _logger.debug(f"Chunk {timestamp} has similarity {distance:.4f} with '{filename}'")
                    
                    if distance < threshold:
                        detection = {
                            "timestamp": timestamp,
                            "wakeword": filename,
                            "distance": distance
                        }
                        _logger.info(f"Wake word '{filename}' detected with distance {distance:.4f}")
                        callback(detection, stream)
                    
                except Exception as e:
                    _logger.error(f"DTW comparison failed for {filename}: {e}")