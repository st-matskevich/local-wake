import argparse
import logging
from features import extract_mfcc_features, extract_embedding_features, dtw_cosine_normalized_distance

_logger = logging.getLogger("local-wake")

def main():
    parser = argparse.ArgumentParser(description="Compare two audio files features via DTW.")
    parser.add_argument("audio1", type=str, 
                        help="Path to first audio file"
    )
    parser.add_argument("audio2", type=str, 
                        help="Path to second audio file"
    )
    parser.add_argument("--method", choices=["mfcc", "embedding"], default="embedding", 
                        help="Feature extraction method (default: embedding)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    _logger.info(f"Comparing {args.audio1} and {args.audio2} using {args.method} features...")
    
    if args.method == "mfcc":
        features1 = extract_mfcc_features(args.audio1)
        features2 = extract_mfcc_features(args.audio2)
    else:  # embedding
        features1 = extract_embedding_features(args.audio1)
        features2 = extract_embedding_features(args.audio2)
    
    _logger.info(f"Features 1 shape: {features1.shape}")
    _logger.info(f"Features 2 shape: {features2.shape}")
    
    distance = dtw_cosine_normalized_distance(features1, features2)
    _logger.info(f"Normalized DTW distance (cosine): {distance:.4f}")

if __name__ == "__main__":
    main()