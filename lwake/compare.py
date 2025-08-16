import logging

_logger = logging.getLogger("local-wake")

def compare(audio1, audio2, method="embedding"):
    """Compare two audio files using specified method"""
    from .features import extract_mfcc_features, extract_embedding_features, dtw_cosine_normalized_distance
    
    _logger.info(f"Comparing {audio1} and {audio2} using {method} features...")
    
    if method == "mfcc":
        features1 = extract_mfcc_features(audio1)
        features2 = extract_mfcc_features(audio2)
    else:  # embedding
        features1 = extract_embedding_features(audio1)
        features2 = extract_embedding_features(audio2)
    
    _logger.info(f"Features 1 shape: {features1.shape}")
    _logger.info(f"Features 2 shape: {features2.shape}")
    
    distance = dtw_cosine_normalized_distance(features1, features2)
    _logger.info(f"Normalized DTW distance (cosine): {distance:.4f}")
    
    return distance
