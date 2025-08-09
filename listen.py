import os
import queue
import time
import numpy as np
import sounddevice as sd
import librosa
import argparse

import soundfile as sf

def extract_features(
    path=None,
    y=None,
    sr=16000,
    n_mfcc=13,
    frame_length=400,
    hop_length=160,
):
    if y is None and path is None:
        raise ValueError("Must provide either path or raw audio y")
    if y is None:
        y, _ = librosa.load(path, sr=sr)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=512,
        hop_length=hop_length,
        win_length=frame_length,
        window="hann"
    )

    return mfcc

def dtw_cosine_normalized_distance(mfcc1, mfcc2):
    cost_matrix, _ = librosa.sequence.dtw(X=mfcc1, Y=mfcc2, metric='cosine')
    total_cost = cost_matrix[-1, -1]
    normalization = mfcc1.shape[0] + mfcc2.shape[0]
    return total_cost / normalization

def load_support_set(support_folder):
    support = []
    for file in os.listdir(support_folder):
        if not file.endswith(".wav"):
            continue
        path = os.path.join(support_folder, file)
        mfcc = extract_features(path=path)
        if mfcc is not None:
            support.append((file, mfcc))
    return support

def main():
    parser = argparse.ArgumentParser(description="Real-time wake word detection")
    parser.add_argument("support_folder", type=str, help="Folder with reference wake word .wav files")
    parser.add_argument("threshold", type=float, help="DTW cosine distance threshold for detection")
    args = parser.parse_args()

    support_set = load_support_set(args.support_folder)
    if not support_set:
        print("[ERROR] No valid wake word files found in support folder")
        return

    import sounddevice as sd
    import queue

    buffer_size = 2 * 16000  # 1 second at 16kHz
    slide_size = 8000    # 0.5 second slide

    audio_buffer = np.zeros(buffer_size, dtype=np.float32)
    q = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[WARN] Audio input status: {status}")
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=16000, channels=1, blocksize=slide_size, callback=audio_callback):
        print("[INFO] Listening for wake words...")
        while True:
            try:
                chunk = q.get(timeout=1)
            except queue.Empty:
                continue
            audio_buffer = np.roll(audio_buffer, -len(chunk))
            audio_buffer[-len(chunk):] = chunk

            # save audio for debug
            timestamp = int(time.time() * 1000)
            # filename = f"debug_audio_{timestamp}.wav"
            # sf.write(filename, audio_buffer, samplerate=16000)

            mfcc = extract_features(y=audio_buffer)
            if mfcc is None:
                continue
            for filename, ref_mfcc in support_set:
                dist = dtw_cosine_normalized_distance(mfcc, ref_mfcc)
                print(f"[TRACE] Chunk {timestamp} has similarity {dist:.4f} with '{filename}'")
                if dist < args.threshold:
                    print(f"[DETECTED] Wake word '{filename}' detected with distance {dist:.4f} at chunk {timestamp}")

if __name__ == "__main__":
    main()
