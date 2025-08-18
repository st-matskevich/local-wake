import librosa
import onnxruntime as ort
import functools
import logging
from importlib.resources import files

_logger = logging.getLogger("local-wake")

@functools.cache
def _load_embedding_model():
    """Load the speech_embedding model (cached)"""
    _logger.info("Loading speech_embedding model...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1 
    model_file = files('lwake.models') / 'speech-embedding.onnx'
    session = ort.InferenceSession(model_file, opts, providers=["CPUExecutionProvider"])
    _logger.info("Model loaded successfully")
    return session

def extract_mfcc_features(
    path=None,
    y=None,
    sample_rate=16000,
    n_mfcc=13,
    frame_length=400,
    hop_length=160,
):
    """Extract speech features using MFCC from audio file or raw audio"""
    if y is None and path is None:
        raise ValueError("Must provide either path or raw audio y")
    
    if y is None:
        y, _ = librosa.load(path, sr=sample_rate)
    
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sample_rate,
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
    sample_rate=16000,
):
    """Extract audio features using speech_embedding from audio file or raw audio"""
    if y is None and path is None:
        raise ValueError("Must provide either path or raw audio y")
    
    if y is None:
        y, _ = librosa.load(path, sr=sample_rate)
    
    model = _load_embedding_model()
    emb = model.run(None,  {"samples:0": y[None, :]})[0]
    
    # Return transposed shape (96, time_frames) for DTW
    return emb[0, :, 0, :].T

def dtw_cosine_normalized_distance(features1, features2):
    """Compute normalized DTW distance with cosine metric"""
    cost_matrix, _ = librosa.sequence.dtw(X=features1, Y=features2, metric='cosine')
    total_cost = cost_matrix[-1, -1]
    
    # Normalize by sum of sequence lengths (time dimension)
    normalization = features1.shape[1] + features2.shape[1]
    return total_cost / normalization