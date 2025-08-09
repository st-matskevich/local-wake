# listen.py
import os
import queue
import time
import numpy as np
import sounddevice as sd
import argparse
from features import extract_mfcc_features, extract_embedding_features, dtw_cosine_normalized_distance

def load_support_set(support_folder, method="embedding"):
    """Load reference wake word files from support folder"""
    support = []
    
    for file in os.listdir(support_folder):
        if not file.endswith(".wav"):
            continue
        
        path = os.path.join(support_folder, file)
        print(f"Loading reference file: {file}")
        
        try:
            if method == "mfcc":
                features = extract_mfcc_features(path=path)
            else:  # embedding
                features = extract_embedding_features(path=path)
            
            if features is not None:
                support.append((file, features))
                print(f"  Features shape: {features.shape}")
        except Exception as e:
            print(f"  Error loading {file}: {e}")
    
    return support

def main():
    parser = argparse.ArgumentParser(description="Real-time wake word detection")
    parser.add_argument("support_folder", type=str, help="Folder with reference wake word .wav files")
    parser.add_argument("threshold", type=float, help="DTW cosine distance threshold for detection")
    parser.add_argument("--method", choices=["mfcc", "embedding"], default="embedding", 
                       help="Feature extraction method (default: embedding)")
    parser.add_argument("--buffer-size", type=float, default=2.0, 
                       help="Audio buffer size in seconds (default: 2.0)")
    parser.add_argument("--slide-size", type=float, default=0.5, 
                       help="Slide size in seconds (default: 0.5)")
    
    args = parser.parse_args()
    
    print(f"Loading support set using {args.method} features...")
    support_set = load_support_set(args.support_folder, method=args.method)
    
    if not support_set:
        print("[ERROR] No valid wake word files found in support folder")
        return
    
    print(f"Loaded {len(support_set)} reference files")
    
    # Audio streaming setup
    sr = 16000
    buffer_size = int(args.buffer_size * sr)
    slide_size = int(args.slide_size * sr)
    
    audio_buffer = np.zeros(buffer_size, dtype=np.float32)
    q = queue.Queue()
    
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[WARN] Audio input status: {status}")
        q.put(indata[:, 0].copy())
    
    print(f"[INFO] Starting audio stream (buffer: {args.buffer_size}s, slide: {args.slide_size}s)")
    print(f"[INFO] Using {args.method} features with threshold {args.threshold}")
    print("[INFO] Listening for wake words...")
    
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
                print(f"[ERROR] Feature extraction failed: {e}")
                continue
            
            # Compare with reference files
            timestamp = int(time.time() * 1000)
            
            for filename, ref_features in support_set:
                try:
                    distance = dtw_cosine_normalized_distance(features, ref_features)
                    
                    if distance < args.threshold:
                        print(f"[DETECTED] Wake word '{filename}' detected with distance {distance:.4f} at {timestamp}")
                    
                except Exception as e:
                    print(f"[ERROR] DTW comparison failed for {filename}: {e}")

if __name__ == "__main__":
    main()