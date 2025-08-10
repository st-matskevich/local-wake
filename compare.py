import argparse
from features import extract_mfcc_features, extract_embedding_features, dtw_cosine_normalized_distance

def main():
    parser = argparse.ArgumentParser(description="Compare two audio files features via DTW.")
    parser.add_argument("audio1", type=str, help="Path to first audio file")
    parser.add_argument("audio2", type=str, help="Path to second audio file")
    parser.add_argument("--method", choices=["mfcc", "embedding"], default="embedding", 
                       help="Feature extraction method (default: embedding)")
    
    args = parser.parse_args()
    
    print(f"Comparing {args.audio1} and {args.audio2} using {args.method} features...")
    
    if args.method == "mfcc":
        features1 = extract_mfcc_features(args.audio1)
        features2 = extract_mfcc_features(args.audio2)
    else:  # embedding
        features1 = extract_embedding_features(args.audio1)
        features2 = extract_embedding_features(args.audio2)
    
    print(f"Features 1 shape: {features1.shape}")
    print(f"Features 2 shape: {features2.shape}")
    
    distance = dtw_cosine_normalized_distance(features1, features2)
    print(f"Normalized DTW distance (cosine): {distance:.4f}")

if __name__ == "__main__":
    main()