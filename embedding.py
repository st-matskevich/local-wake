import tensorflow_hub as hub
import librosa
import numpy as np
import tensorflow as tf

def dtw_cosine_normalized_distance(mfcc1, mfcc2):
    cost_matrix, _ = librosa.sequence.dtw(X=mfcc1, Y=mfcc2, metric='cosine')
    total_cost = cost_matrix[-1, -1]
    normalization = mfcc1.shape[0] + mfcc2.shape[0]
    return total_cost / normalization

# Load the model
model = hub.load("https://tfhub.dev/google/speech_embedding/1")
inference_fn = model.signatures['default']

# Load audio and check actual duration
audio1, _ = librosa.load("./recording/sample-1.wav", sr=16000)
audio2, _ = librosa.load("./bad-1.wav", sr=16000)

# Get embeddings
embeddings1 = inference_fn(default=tf.constant(audio1[None, :], dtype=tf.float32))['default']
embeddings2 = inference_fn(default=tf.constant(audio2[None, :], dtype=tf.float32))['default']

emb1 = embeddings1[0, :, 0, :].numpy().T
emb2 = embeddings2[0, :, 0, :].numpy().T

print(f"shape{emb1.shape}")

distance = dtw_cosine_normalized_distance(emb1, emb2)
print(f"Normalized DTW distance (cosine): {distance}")
