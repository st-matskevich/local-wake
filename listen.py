import os
import sys
import queue
import time
import json
import numpy as np
import sounddevice as sd
import argparse
import logging
from features import extract_mfcc_features, extract_embedding_features, dtw_cosine_normalized_distance

_logger = logging.getLogger("local-wake")

def load_support_set(support_folder, method="embedding"):
    """Load reference wake word files from support folder"""
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

def main():
    parser = argparse.ArgumentParser(description="Real-time wake word detection")
    parser.add_argument("support_folder", type=str, 
                        help="Folder with reference wake word .wav files"
    )
    parser.add_argument("threshold", type=float, 
                        help="DTW cosine distance threshold for detection"
    )
    parser.add_argument("--method", choices=["mfcc", "embedding"], default="embedding", 
                        help="Feature extraction method (default: embedding)"
    )
    parser.add_argument("--buffer-size", type=float, default=2.0, 
                        help="Audio buffer size in seconds (default: 2.0)"
    )
    parser.add_argument("--slide-size", type=float, default=0.25, 
                        help="Slide size in seconds (default: 0.25)"
    )
    parser.add_argument("--debug", action="store_true", 
                        help="Print debug messages to the stderr"
    )
    args = parser.parse_args()
 
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG
    
    logging.basicConfig(level=level)
    
    _logger.info(f"Loading support set using {args.method} features...")
    support_set = load_support_set(args.support_folder, method=args.method)
    
    if not support_set:
        _logger.error("No valid wake word files found in support folder")
        return
    
    _logger.info(f"Loaded {len(support_set)} reference files")
    
    # Audio streaming setup
    sr = 16000
    buffer_size = int(args.buffer_size * sr)
    slide_size = int(args.slide_size * sr)
    
    audio_buffer = np.zeros(buffer_size, dtype=np.float32)
    q = queue.Queue()
    
    def audio_callback(indata, frames, time_info, status):
        if status:
            _logger.warning(f"Audio input status: {status}")
        q.put(indata[:, 0].copy())
    
    _logger.info(f"Starting audio stream (buffer: {args.buffer_size}s, slide: {args.slide_size}s)")
    _logger.info(f"Using {args.method} features with threshold {args.threshold}")
    _logger.info("Listening for wake words...")
    
    with sd.InputStream(samplerate=sr, channels=1, blocksize=slide_size, callback=audio_callback):
        while True:
            try:
                chunk = q.get(timeout=1)
            except queue.Empty:
                continue
            
            # Update sliding buffer
            audio_buffer = np.roll(audio_buffer, -len(chunk))
            audio_buffer[-len(chunk):] = chunk
            
            # Extract features from current buffer
            try:
                if args.method == "mfcc":
                    features = extract_mfcc_features(y=audio_buffer, sr=sr)
                else:  # embedding
                    features = extract_embedding_features(y=audio_buffer, sr=sr)
                
                if features is None:
                    continue
                
            except Exception as e:
                _logger.error(f"Feature extraction failed: {e}")
                continue
            
            # Compare with reference files
            timestamp = int(time.time() * 1000)
            
            for filename, ref_features in support_set:
                try:
                    distance = dtw_cosine_normalized_distance(features, ref_features)
                    _logger.debug(f"Chunk {timestamp} has similarity {distance:.4f} with '{filename}'")
                    
                    if distance < args.threshold:
                        detection = {
                            "timestamp": timestamp,
                            "wakeword": filename,
                            "distance": distance
                        }
                        _logger.info(f"Wake word '{filename}' detected with distance {distance:.4f}")
                        print(json.dumps(detection), file=sys.stdout, flush=True)
                    
                except Exception as e:
                    _logger.error(f"DTW comparison failed for {filename}: {e}")

if __name__ == "__main__":
    main()