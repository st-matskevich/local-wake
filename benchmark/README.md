## Benchmark

This benchmark evaluates the performance of the [local-wake](../README.md). Tests are designed to test how well the system performs under realistic conditions, including cross-speaker variation and audio augmentations such as noise and speed changes.

### Dataset

The benchmark uses the [Qualcomm Keyword Speech Dataset (2019)](https://www.qualcomm.com/developer/software/keyword-speech-dataset), which contains recordings of various keywords from multiple speakers. This dataset is for research use only. You must obtain it separately from Qualcomm as it cannot be redistributed with this code.


### Test Setup
The benchmark splits each keyword/speaker combination into:
- **Reference set**: First `REFERENCE_SET_SIZE` recordings (default: 3) used as templates
- **Test samples**: Remaining recordings used for positive/negative evaluation

### Evaluation

The benchmark implements two complementary tests, each executed in two variants:

#### Test Types:
1. **ROC Analysis** - Determines optimal thresholds.
2. **Accuracy Test** - Evaluates system performance at optimal thresholds.

#### Test Variants:

1. **Same-speaker test** - Speaker-dependent keyword detection:
    - Positives: Same speaker, same keyword (excluding reference set)
    - Negatives: Any speaker, different keywords
    - Purpose: Measures detection for enrolled speakers

1. **Cross-speaker test** - Speaker-independent keyword detection:
    - Positives: Any speaker, same keyword (excluding reference speaker)  
    - Negatives: Any speaker, different keywords
    - Purpose: Measures generalization across speakers

#### Audio Augmentations

Optional augmentations simulate real-world conditions:

1. **Speed variation**: ±25% speed changes (0.75x - 1.25x) to simulate natural speech rate differences
2. **Background noise**: White noise mixed at 25% volume relative to speech RMS
3. **Temporal misalignment**: Random padding up to 2000 samples at start/end to test alignment robustness


### Results
| Condition | Test Type | Positive Accuracy | Negative Accuracy | Optimal Threshold | AUC | 
|-----------|-----------|------------------|------------------|-----|-------------------|
| Clean Audio | Same-speaker | 98.6% | 99.7% | 0.1623 | 0.9967 | 
| | Cross-speaker | 81.9% | 93.2% |  0.2075 | 0.9361 |
| With Augmentation | Same-speaker | 94.0% | 98.6% | 0.1405 | 0.9894 |
| | Cross-speaker | 72.6% | 89.7% | 0.1685 | 0.8745 |

### Re-evaluation
You can manually re-evaluate the benchmark to confirm results, change settings or use with a different dataset. To do so:
```
python benchmark.py
```

There are some configuration values available:
- `DATASET_PATH` (default: benchmark/qualcomm) - Path to Qualcomm dataset
- `ENABLE_AUGMENTATION` (default: False) - Enable audio augmentations 
- `REFERENCE_SET_SIZE` (default: 3) - Number of reference recordings per speaker 
- `SPEED_LEVEL` (default: 0.25) - Speed variation range
- `NOISE_LEVEL` (default: 0.25) - Background noise level (25% of speech RMS)
- `MAX_RANDOM_PAD` (default: 2000) - Maximum random padding samples

The benchmark will produce some artifacts for detailed analysis:
- CSV with per-speaker evaluation results stored at `RESULT_CSV` (default: benchmark/result.csv)
- A subset of recordings with augmentation applied for manual verification stored at `EXPORT_AUGMENTATION` (default: benchmark/augmented)

### To do
 - While some Qualcomm samples are faint or inaudible, the data set is used as it is. Consider excluding them from the benchmark for a better representability.