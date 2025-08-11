import librosa
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import logging

_model = None
_inference_fn = None
_logger = logging.getLogger("local-wake")

def _load_embedding_model():
    """Load the speech embedding model (lazy loading)"""
    global _model, _inference_fn
    if _model is None:
        _logger.info("Loading speech embedding model...")
        _model = hub.load("https://tfhub.dev/google/speech_embedding/1")
        _inference_fn = _model.signatures['default']
        _logger.info("Model loaded successfully")
    return _inference_fn

def extract_mfcc_features(
    path=None,
    y=None,
    sr=16000,
    n_mfcc=13,
    frame_length=400,
    hop_length=160,
):
    """Extract MFCC features from audio file or raw audio"""
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

def extract_embedding_features(
    path=None,
    y=None,
    sr=16000,
):
    """Extract speech embedding features from audio file or raw audio"""
    if y is None and path is None:
        raise ValueError("Must provide either path or raw audio y")
    
    if y is None:
        y, _ = librosa.load(path, sr=sr)
    
    inference_fn = _load_embedding_model()
    emb = inference_fn(default=tf.constant(y[None, :], dtype=tf.float32))['default']
    
    # Return transposed shape (96, time_frames) for DTW
    return emb[0, :, 0, :].numpy().T

def dtw_cosine_normalized_distance(features1, features2):
    """Compute normalized DTW distance with cosine metric"""
    cost_matrix, _ = librosa.sequence.dtw(X=features1, Y=features2, metric='cosine')
    total_cost = cost_matrix[-1, -1]
    
    # Normalize by sum of sequence lengths (time dimension)
    normalization = features1.shape[1] + features2.shape[1]
    return total_cost / normalization