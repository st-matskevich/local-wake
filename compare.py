import librosa
import argparse

def extract_features(
    path,
    sr=16000,
    n_mfcc=13,
    frame_length=400,
    hop_length=160,
):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two audio files using MFCC + DTW.")
    parser.add_argument("audio1", type=str, help="Path to first audio file")
    parser.add_argument("audio2", type=str, help="Path to second audio file")
    args = parser.parse_args()

    mfcc1 = extract_features(args.audio1)
    mfcc2 = extract_features(args.audio2)

    distance = dtw_cosine_normalized_distance(mfcc1, mfcc2)
    print(f"Normalized DTW distance (cosine): {distance}")
