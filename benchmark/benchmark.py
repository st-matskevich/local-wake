import os
import numpy as np
import soundfile as sf
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from scipy import signal
from lwake.features import extract_embedding_features, dtw_cosine_normalized_distance

# ==================== CONFIGURATION ====================
DATASET_PATH = "benchmark/qualcomm"
RESULT_CSV = "benchmark/result.csv"
REFERENCE_SET_SIZE = 3
MIN_SAMPLES = 12400

ENABLE_AUGMENTATION = False
EXPORT_AUGMENTATION = "benchmark/augmented"
RANDOM_SEED = 1337
SPEED_LEVEL = 0.25  # 25% slower or faster
NOISE_LEVEL = 0.25  # Noise volume relative to audio (25%)
MAX_RANDOM_PAD = 2000  # Maximum random padding samples
# ========================================================

def apply_speed_change(audio, speed_factor):
    """Apply speed change to audio using resampling."""
    if speed_factor == 1.0:
        return audio
    
    new_length = int(len(audio) / speed_factor)
    resampled = signal.resample(audio, new_length)
    return resampled

def generate_noise(length):
    """Generate synthetic noise."""
    return np.random.normal(0, 1, length)

def add_background_noise(audio, noise_level=0.25):
    """Add background noise to audio."""
    if noise_level <= 0:
        return audio
    
    noise = generate_noise(len(audio))
    audio_rms = np.sqrt(np.mean(audio**2))
    if audio_rms > 0:
        noise_rms = np.sqrt(np.mean(noise**2))
        noise = noise * (audio_rms * noise_level / noise_rms)
    
    return audio + noise

def add_random_padding(audio, max_pad_samples):
    """Add random padding at start and/or end."""
    if max_pad_samples <= 0:
        return audio
    
    # Random padding amounts
    pad_start = np.random.randint(0, max_pad_samples // 2 + 1)
    pad_end = np.random.randint(0, max_pad_samples // 2 + 1)
    
    if pad_start == 0 and pad_end == 0:
        return audio
    
    return np.pad(audio, (pad_start, pad_end), mode='constant')

def apply_augmentations(audio):
    """Apply all augmentations to audio sample."""
    if not ENABLE_AUGMENTATION:
        return audio
    
    # 1. Apply random speed change
    speed_factor = np.random.uniform(1 - SPEED_LEVEL, 1 + SPEED_LEVEL)
    audio = apply_speed_change(audio, speed_factor)
    
    # 2. Add background noise
    audio = add_background_noise(audio, NOISE_LEVEL)
    
    # 3. Add random temporal misalignment
    audio = add_random_padding(audio, MAX_RANDOM_PAD)
    
    return audio

def pad_audio(audio, target_length):
    """Pad audio equally on both sides to reach target length."""
    if len(audio) >= target_length:
        return audio
    
    pad_total = target_length - len(audio)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(audio, (pad_left, pad_right), mode='constant')

def load_and_preprocess_dataset(dataset_path):
    """Load dataset and extract features into nested dictionary."""
    features = {}
    np.random.seed(RANDOM_SEED)
    
    for keyword_dir in Path(dataset_path).iterdir():
        if not keyword_dir.is_dir():
            continue
            
        keyword = keyword_dir.name
        features[keyword] = {}
        
        for speaker_dir in keyword_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
                
            speaker = speaker_dir.name
            features[keyword][speaker] = {'reference': {}, 'positive': {}}
            
            # Load all recordings for this speaker/keyword
            recordings = []
            for wav_file in speaker_dir.glob("*.wav"):
                audio, _ = sf.read(wav_file)
                
                # Apply augmentations
                audio = apply_augmentations(audio)
                audio = pad_audio(audio, MIN_SAMPLES)
                audio = audio.astype(np.float32, copy=False)

                feat = extract_embedding_features(y=audio)
                recordings.append((wav_file.name, feat))
            
            # Split into reference and positive sets
            recordings.sort(key=lambda x: x[0])  # Sort for reproducibility
            
            for i, (filename, feat) in enumerate(recordings):
                if i < REFERENCE_SET_SIZE:
                    features[keyword][speaker]['reference'][filename] = feat
                else:
                    features[keyword][speaker]['positive'][filename] = feat
    
    return features

def get_min_distance_to_references(test_feat, reference_features):
    """Get minimum distance from test feature to any reference feature."""
    distances = []
    for ref_feat in reference_features.values():
        distance = dtw_cosine_normalized_distance(test_feat, ref_feat)
        distances.append(distance)
    return min(distances) if distances else float('inf')

def collect_all_distances(features):
    """Collect all distances organized by keyword/speaker/test_type."""
    distances = {}
    
    for keyword in features:
        distances[keyword] = {}
        
        for speaker in features[keyword]:
            reference_features = features[keyword][speaker]['reference']
            
            if not reference_features:  # Skip if no references
                continue
                
            distances[keyword][speaker] = {
                'same_pos': [],    # Same speaker, same keyword positives
                'cross_pos': [],   # Different speaker, same keyword positives
                'neg': []          # Different keyword negatives
            }
            
            # Same-speaker positives (same speaker, same keyword)
            positive_features = features[keyword][speaker]['positive']
            for feat in positive_features.values():
                dist = get_min_distance_to_references(feat, reference_features)
                distances[keyword][speaker]['same_pos'].append(dist)
            
            # Cross-speaker positives (different speaker, same keyword)
            for other_speaker in features[keyword]:
                if other_speaker != speaker:
                    positive_features = features[keyword][other_speaker]['positive']
                    for feat in positive_features.values():
                        dist = get_min_distance_to_references(feat, reference_features)
                        distances[keyword][speaker]['cross_pos'].append(dist)
            
            # Negatives (other keywords, any speaker)
            for other_keyword in features:
                if other_keyword != keyword:
                    for other_speaker in features[other_keyword]:
                        negative_features = features[other_keyword][other_speaker]['positive']
                        for feat in negative_features.values():
                            dist = get_min_distance_to_references(feat, reference_features)
                            distances[keyword][speaker]['neg'].append(dist)
    
    return distances

def distances_to_roc_data(distances, same_speaker=True):
    """Convert distance map to arrays for ROC analysis."""
    all_distances = []
    all_labels = []
    
    for keyword in distances:
        for speaker in distances[keyword]:
            speaker_data = distances[keyword][speaker]
            
            # Positive samples
            if same_speaker:
                pos_distances = speaker_data['same_pos']
            else:
                pos_distances = speaker_data['cross_pos']
            
            all_distances.extend(pos_distances)
            all_labels.extend([1] * len(pos_distances))
            
            # Negative samples
            neg_distances = speaker_data['neg']
            all_distances.extend(neg_distances)
            all_labels.extend([0] * len(neg_distances))
    
    return np.array(all_distances), np.array(all_labels)

def find_optimal_threshold(distances, labels):
    """Find optimal threshold using Youden's index."""
    fpr, tpr, thresholds = roc_curve(labels, -distances)  # Negative because smaller distance = more similar
    youden_scores = tpr - fpr
    optimal_idx = np.argmax(youden_scores)
    optimal_threshold = thresholds[optimal_idx]
    roc_auc = auc(fpr, tpr)
    return -optimal_threshold, roc_auc  # Convert back to positive

def evaluate_accuracy_from_distances(distances, threshold, same_speaker=True):
    """Evaluate accuracy using given threshold from distance map."""
    results = {}
    
    for keyword in distances:
        results[keyword] = {}
        
        for speaker in distances[keyword]:
            speaker_data = distances[keyword][speaker]
            
            # Positive samples
            if same_speaker:
                pos_distances = speaker_data['same_pos']
            else:
                pos_distances = speaker_data['cross_pos']
            
            pos_correct = sum(1 for d in pos_distances if d < threshold)
            pos_total = len(pos_distances)
            
            # Negative samples
            neg_distances = speaker_data['neg']
            neg_correct = sum(1 for d in neg_distances if d >= threshold)
            neg_total = len(neg_distances)
            
            results[keyword][speaker] = {
                'positive_correct': pos_correct,
                'negative_correct': neg_correct,
                'positive_samples': pos_total,
                'negative_samples': neg_total
            }
    
    return results

def print_results(results, test_name):
    """Print accuracy results."""
    total_pos_correct = total_pos_samples = 0
    total_neg_correct = total_neg_samples = 0
    
    for keyword in results:
        for speaker in results[keyword]:
            r = results[keyword][speaker]
            total_pos_correct += r['positive_correct']
            total_pos_samples += r['positive_samples']
            total_neg_correct += r['negative_correct']
            total_neg_samples += r['negative_samples']

    overall_pos_acc = total_pos_correct / total_pos_samples if total_pos_samples > 0 else 0
    overall_neg_acc = total_neg_correct / total_neg_samples if total_neg_samples > 0 else 0

    print(f"\n{test_name} Results:")
    print("=" * 50)
    print(f"Positive Accuracy: {overall_pos_acc:.3f} ({total_pos_samples} samples)")
    print(f"Negative Accuracy: {overall_neg_acc:.3f} ({total_neg_samples} samples)")

def export_results_to_csv(same_results, cross_results, filename):
    """Export detailed per-speaker results to CSV."""
    import csv
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['test_type', 'keyword', 'speaker', 'positive_accuracy', 'positive_correct', 'positive_samples', 
                     'negative_accuracy', 'negative_correct', 'negative_samples']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        # Write same-speaker results
        for keyword in same_results:
            for speaker in same_results[keyword]:
                r = same_results[keyword][speaker]
                pos_acc = r['positive_correct'] / r['positive_samples'] if r['positive_samples'] > 0 else 0
                neg_acc = r['negative_correct'] / r['negative_samples'] if r['negative_samples'] > 0 else 0
                
                writer.writerow({
                    'test_type': 'same-speaker',
                    'keyword': keyword,
                    'speaker': speaker,
                    'positive_accuracy': f"{pos_acc:.3f}",
                    'positive_correct': r['positive_correct'],
                    'positive_samples': r['positive_samples'],
                    'negative_accuracy': f"{neg_acc:.3f}",
                    'negative_correct': r['negative_correct'],
                    'negative_samples': r['negative_samples']
                })
        
        # Write cross-speaker results
        for keyword in cross_results:
            for speaker in cross_results[keyword]:
                r = cross_results[keyword][speaker]
                pos_acc = r['positive_correct'] / r['positive_samples'] if r['positive_samples'] > 0 else 0
                neg_acc = r['negative_correct'] / r['negative_samples'] if r['negative_samples'] > 0 else 0
                
                writer.writerow({
                    'test_type': 'cross-speaker',
                    'keyword': keyword,
                    'speaker': speaker,
                    'positive_accuracy': f"{pos_acc:.3f}",
                    'positive_correct': r['positive_correct'],
                    'positive_samples': r['positive_samples'],
                    'negative_accuracy': f"{neg_acc:.3f}",
                    'negative_correct': r['negative_correct'],
                    'negative_samples': r['negative_samples']
                })
    
    print(f"Detailed results exported to: {filename}")

def export_augmented_audio(dataset_path, output_dir,
                           max_speakers_per_keyword=5, max_audio_per_speaker=3):
    """
    Export augmented audio for manual listening:
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for keyword_dir in Path(dataset_path).iterdir():
        if not keyword_dir.is_dir():
            continue
        keyword = keyword_dir.name
        keyword_output = output_dir / keyword
        keyword_output.mkdir(exist_ok=True)

        speakers = list(keyword_dir.iterdir())
        for speaker_dir in speakers[:max_speakers_per_keyword]:
            if not speaker_dir.is_dir():
                continue
            speaker = speaker_dir.name
            speaker_output = keyword_output / speaker
            speaker_output.mkdir(exist_ok=True)

            wav_files = list(speaker_dir.glob("*.wav"))
            for wav_file in wav_files[:max_audio_per_speaker]:
                audio, sr = sf.read(wav_file)
                augmented_audio = apply_augmentations(audio)
                augmented_audio = pad_audio(augmented_audio, MIN_SAMPLES)

                out_file = speaker_output / f"aug_{wav_file.name}"
                sf.write(out_file, augmented_audio, sr)
                print(f"Exported {out_file}")

def main():
    print("Loading and preprocessing dataset...")
    if ENABLE_AUGMENTATION:
        print(f"Augmentation enabled: speed={SPEED_LEVEL}, noise={NOISE_LEVEL}, padding={MAX_RANDOM_PAD}")
        export_augmented_audio(DATASET_PATH, EXPORT_AUGMENTATION)
        
    features = load_and_preprocess_dataset(DATASET_PATH)
    
    print("Collecting all distances...")
    distances = collect_all_distances(features)
    
    # ROC Analysis
    print("\n" + "="*60)
    print("ROC ANALYSIS")
    print("="*60)
    
    # Same-speaker ROC
    print("\nSame-speaker ROC analysis...")
    distances_same, labels_same = distances_to_roc_data(distances, same_speaker=True)
    threshold_same, auc_same = find_optimal_threshold(distances_same, labels_same)
    print(f"Same-speaker - Optimal threshold: {threshold_same:.4f}, AUC: {auc_same:.4f}")
    
    # Cross-speaker ROC
    print("\nCross-speaker ROC analysis...")
    distances_cross, labels_cross = distances_to_roc_data(distances, same_speaker=False)
    threshold_cross, auc_cross = find_optimal_threshold(distances_cross, labels_cross)
    print(f"Cross-speaker - Optimal threshold: {threshold_cross:.4f}, AUC: {auc_cross:.4f}")
    
    # Accuracy Tests
    print("\n" + "="*60)
    print("ACCURACY TESTS")
    print("="*60)
    print(f"Using {threshold_same:.4f} as same-speaker threshold and {threshold_cross:.4f} as cross-speaker threshold ")
    
    # Same-speaker accuracy
    same_results = evaluate_accuracy_from_distances(distances, threshold_same, same_speaker=True)
    print_results(same_results, "Same-speaker Accuracy")
    
    # Cross-speaker accuracy
    cross_results = evaluate_accuracy_from_distances(distances, threshold_cross, same_speaker=False)
    print_results(cross_results, "Cross-speaker Accuracy")
    
    # Export detailed results to CSV
    export_results_to_csv(same_results, cross_results, RESULT_CSV)

if __name__ == "__main__":
    main()