# 🎙️ Speech Emotion Recognition

> Exploring how different neural network architectures handle temporal audio signals — from flat feature vectors to full sequence models.

---

## Overview

This project investigates speech emotion classification using the [RAVDESS dataset](https://zenodo.org/records/1188976) (24 actors, 8 emotions). Each architecture is implemented from scratch in PyTorch to build intuition about what temporal context buys you — and what it costs.

The pipeline goes from raw `.wav` → handcrafted features (MFCCs, deltas, ZCR, RMS) → three progressively more powerful models.

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

**Split strategy:** speaker-independent, gender-balanced
- Train: actors 1–16 (8M + 8F)
- Val: actors 17–20 (2M + 2F)  
- Test: actors 21–24 (2M + 2F) — held out until final evaluation

---

## Features

Extracted per clip using `librosa`, saved as `.npz` (one file per clip):

| Feature | Shape | Captures |
|---|---|---|
| MFCC | `(40, T)` | Vocal tract shape / timbre |
| MFCC Δ | `(40, T)` | Rate of spectral change |
| MFCC ΔΔ | `(40, T)` | Acceleration of spectral change |
| ZCR | `(1, T)` | Voicing / breathiness |
| RMS | `(1, T)` | Energy / loudness |

---

## Models

### MLP — Baseline
> No temporal context. Features averaged over time → flat 122-dim vector.

The classic feedforward network. Fast to train, easy to interpret, but fundamentally limited — averaging over time destroys the prosodic trajectory that carries most of the emotional signal.

```
122 → 256 → 128 → 64 → 8
```

| | Value |
|---|---|
| Val accuracy | ~32% |
| Angry F1 | 0.65 |
| Disgust F1 | 0.05 |
| Optimizer | Adam |
| Dropout | 0.3 |

**Takeaway:** angry and calm are separable from static features. Disgust collapses — its acoustic signature requires temporal context to distinguish from similar emotions.

---

### CNN — Local Temporal Patterns
> 1D convolutions over MFCC time axis. Learns local spectral-temporal patterns.

Convolutional filters slide over the time axis and learn *where* in the utterance discriminative patterns occur. Captures local prosodic events (onset sharpness, pitch jumps) but still limited in modeling the full temporal arc of an emotion.

```
(82, T) → Conv1d blocks × 3 → Flatten → Linear → 8
```

| | Value |
|---|---|
| Val accuracy | TBD |
| Optimizer | TBD |

---

### LSTM / GRU — Full Sequence Modeling
> Recurrent architecture. Models the entire temporal trajectory frame by frame.

Emotion unfolds over time — the prosodic arc of a full utterance is what humans use to judge emotional state. LSTMs maintain a cell state with additive gradient paths, solving the vanishing gradient problem that makes vanilla RNNs fail on sequences of this length.

```
(82, T) → GRU(hidden=128, layers=2) → final hidden state → Linear → 8
```

| | Value |
|---|---|
| Val accuracy | TBD |
| Optimizer | TBD |

---

## Project Structure

```
.
├── feature_extraction.py        # Extract features from .wav → save .npz
├── data_loader.py               # Load .npz files into records / tensors
├── train_mlp.py                 # MLP training + evaluation
├── train_cnn.py                 # CNN training + evaluation (TBD)
├── train_lstm.py                # LSTM/GRU training + evaluation (TBD)
└── speech_emotion_features.ipynb  # Feature visualization & exploration
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
- Cho et al. (2014) — Learning Phrase Representations using RNN Encoder-Decoder (GRU)
