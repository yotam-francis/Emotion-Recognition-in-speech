# Speech Emotion Recognition

Classifying emotions from speech using the RAVDESS dataset. Three architectures implemented from scratch in PyTorch, each adding more temporal awareness than the last.

---

## Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song
Livingstone & Russo (2018), *PLoS ONE* · [DOI](https://doi.org/10.1371/journal.pone.0196391)

| Property | Value |
|---|---|
| Actors | 24 (12M / 12F) |
| Emotions | neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| Files | 1,440 speech clips |
| Format | 16-bit, 48kHz `.wav` |

**Split:** speaker-independent, gender-balanced
- Train: actors 1-16 (8M + 8F)
- Val: actors 17-20 (2M + 2F)
- Test: actors 21-24 (2M + 2F), held out until final evaluation

---

## Features

Extracted per clip using `librosa`, saved as `.npz`:

| Feature | Shape | Captures |
|---|---|---|
| MFCC | `(40, T)` | Vocal tract shape / timbre |
| MFCC delta | `(40, T)` | Rate of spectral change |
| MFCC delta-delta | `(40, T)` | Acceleration of spectral change |
| ZCR | `(1, T)` | Voicing / breathiness |
| RMS | `(1, T)` | Energy / loudness |

---

## Models

### MLP

Features averaged over time into a flat 122-dim vector. No temporal context.

```
122 -> 256 -> 128 -> 64 -> 8
```

| | Value |
|---|---|
| Val accuracy | ~32% |
| Angry F1 | 0.65 |
| Disgust F1 | 0.05 |
| Optimizer | Adam |
| Dropout | 0.3 |

Angry and calm are reasonably separable from static features. Disgust collapses entirely, its acoustic signature requires temporal context to distinguish from similar emotions.

---

### CNN

1D convolutions over the MFCC time axis. Learns local spectral-temporal patterns without modeling the full sequence.

```
(82, T) -> Conv1d blocks x3 -> Flatten -> Linear -> 8
```

| | Value |
|---|---|
| Val accuracy | TBD |
| Optimizer | TBD |

---

### LSTM / GRU

Recurrent architecture. Processes MFCCs frame by frame, maintaining a hidden state across the full utterance.

```
(82, T) -> GRU(hidden=128, layers=2) -> final hidden state -> Linear -> 8
```

| | Value |
|---|---|
| Val accuracy | TBD |
| Optimizer | TBD |

---

## Project Structure

```
.
├── feature_extraction.py          # Extract features from .wav, save .npz
├── data_loader.py                 # Load .npz files into records / tensors
├── train_mlp.py                   # MLP training and evaluation
├── train_cnn.py                   # CNN training and evaluation (TBD)
├── train_lstm.py                  # LSTM/GRU training and evaluation (TBD)
└── speech_emotion_features.ipynb  # Feature visualization and exploration
```

---

## Setup

```bash
pip install torch librosa soundfile numpy scikit-learn tqdm matplotlib
```

---

## References

- Livingstone SR, Russo FA (2018) — RAVDESS · [PLoS ONE](https://doi.org/10.1371/journal.pone.0196391)
- Hochreiter & Schmidhuber (1997) — Long Short-Term Memory
- Cho et al. (2014) — Learning Phrase Representations using RNN Encoder-Decoder
- Fayek H (2016) — Speech Processing for Machine Learning · [haythamfayek.com](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
- Practical Cryptography — Guide to Mel Frequency Cepstral Coefficients · [practicalcryptography.com](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)
