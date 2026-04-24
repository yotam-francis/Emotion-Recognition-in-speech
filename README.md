# Speech Emotion Recognition

Classifying emotions from speech using the RAVDESS dataset. Four architectures implemented from scratch in PyTorch.

---

## What this project does differently

Most published RAVDESS results (85-97%) use random clip-level splits where the same actor appears in both training and test. This leaks speaker identity into the model. A 2025 subject-independent study confirmed the gap is not small: the same pipeline drops from ~93% to ~36% when switching to a strict actor-disjoint split. The high numbers in the literature are largely speaker recognition results mislabeled as emotion recognition.

This project uses a strict speaker-independent split throughout, gender-balanced so each set has equal male and female representation. The split is fixed by actor, never by clip.

The feature pipeline applies per-clip normalization at extraction time and CMVN per speaker at load time, computed from training actors only with a global fallback for unseen speakers. This removes the speaker-specific spectral offset before the model ever sees the features.

The 2D CNN uses focal loss with class weights instead of standard cross-entropy. Neutral has 96 samples vs 192 for every other class. Under vanilla cross-entropy the minority class collapses into acoustically adjacent classes. Focal loss down-weights easy confidently-predicted samples so hard minority classes keep their gradient signal.

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

**Split:** strict speaker-independent, gender-balanced
- Train: 16 actors (8M + 8F), ~960 clips
- Val: 4 actors (2M + 2F), ~240 clips
- Test: actors 21-24 (2M + 2F), held out until final evaluation

Under this protocol the realistic accuracy ceiling without pretrained models is 70-78%.

---

## Features

Extracted per clip using `librosa` with pre-emphasis (alpha=0.97), saved as `.npz`. Per-clip normalized at extraction. CMVN applied per speaker at load time.

| Feature | Shape | Captures |
|---|---|---|
| MFCC | `(40, T)` | Vocal tract shape / timbre |
| MFCC delta | `(40, T)` | Rate of spectral change |
| MFCC delta-delta | `(40, T)` | Acceleration of spectral change |
| ZCR | `(1, T)` | Voicing / breathiness |
| RMS | `(1, T)` | Energy / loudness |
| Mel spectrogram | `(3, 64, T)` | Full time-frequency representation (mel + delta + delta-delta) |

---

## Models

### MLP

Features averaged over time into a flat 122-dim vector. No temporal context. Serves as a floor for comparison.

```
122 -> 256 -> 128 -> 64 -> 8
```

| | Value |
|---|---|
| Val accuracy | 32% |
| Angry F1 | 0.65 |
| Disgust F1 | 0.05 |
| Optimizer | Adam |
| Dropout | 0.3 |

Angry and calm are separable from static features alone. Disgust collapses entirely since distinguishing it requires temporal context the MLP has no access to.

---

### 1D CNN

1D convolutions over the MFCC time axis. Learns local spectral-temporal patterns. Deltas dropped since conv filters learn temporal derivatives from raw MFCCs directly.

```
(41, T) -> Conv1d blocks x3 -> Global Average Pooling -> Linear -> 8
```

| | Value |
|---|---|
| Val accuracy | 28% |
| Angry F1 | 0.61 |
| Best epoch | 3 |
| Optimizer | Adam |
| Dropout | 0.5 |
| Weight decay | 1e-4 |

Underperformed the MLP. Best weights saved at epoch 3 out of 100, after which validation loss diverged. With ~960 training samples under a strict speaker-independent split, the conv filters had insufficient data to learn patterns that generalize to unseen speakers.

---

### GRU

Recurrent architecture. Processes MFCCs frame by frame and maintains a hidden state across the full utterance. GRU chosen over LSTM for fewer parameters and lower overfitting risk at this dataset size.

```
(300, 41) -> GRU(hidden=64, layers=2) -> last hidden state -> Linear -> 8
```

| | Value |
|---|---|
| Val accuracy | 31% |
| Optimizer | SGD, momentum=0.9 |
| Dropout | 0.3 |
| Noise augmentation | Gaussian std=0.01 |

Matched the MLP despite a more complex architecture. Overfitting onset around epoch 25 regardless of regularization strength.

---

### 2D CNN (depthwise separable)

3-channel mel spectrogram fed into a 2D depthwise-separable CNN. The architecture ranking observed here (2D mel CNN > GRU > 1D CNN) is consistent with published ablations on similar datasets.

```
(3, 64, 300) -> init conv -> depthwise-separable blocks x3 -> AdaptiveAvgPool2d -> Linear -> 8
```

| | Value |
|---|---|
| Val accuracy | 50% |
| Optimizer | Adam |
| Dropout | 0.5 (classifier only) |
| Weight decay | 1e-4 |
| Loss | Focal loss (gamma=2, neutral weight=2.0) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=5) |

Best result across all four models. The jump from 36% to 50% came from switching to focal loss. Neutral and happy recall recovered from near-zero once the dominant classes stopped drowning their gradient. Dropout was removed from conv blocks after observing val loss consistently below train loss, which is a regularization signature rather than underfitting. The model still overfits during training — train and val loss diverge past the best epoch. There is clear room for improvement through better augmentation and architecture changes as described in the ceiling section.

---

## Why the ceiling is where it is

With 16 training actors and a strict speaker-independent split, the bottleneck is speaker variability. ~960 clips from 16 speakers is a limited sample of the space of possible vocal characteristics. Interventions tried with limited effect: CMVN, per-clip normalization, Gaussian noise, weight decay, dropout tuning. The literature points to speed perturbation, reverb augmentation, per-speaker F0 normalization, attention pooling, and multi-scale temporal modeling as the remaining high-leverage changes.

---

## Project Structure

```
.
├── feature_extraction.py          # Extract features from .wav, save .npz
├── data_loader.py                 # Load .npz files, apply CMVN, return tensors
├── train_mlp.ipynb                # MLP training and evaluation
├── train_cnn.ipynb                # 1D CNN training and evaluation
├── train_gru.ipynb                # GRU evaluation
├── GRU.py                         # GRU training script
├── 2DCNN.py                       # 2D CNN training script
└── speech_emotion_features.ipynb  # Feature visualization and exploration
```

---

## Setup

```bash
pip install torch librosa soundfile numpy scikit-learn tqdm matplotlib pyroomacoustics
```

---

## References

- Livingstone SR, Russo FA (2018) — RAVDESS · [PLoS ONE](https://doi.org/10.1371/journal.pone.0196391)
- Majkowski & Kolodziej (2025) — Subject-independent SER on RAVDESS, ~34% macro precision under strict speaker-disjoint 8-class evaluation · [Applied Sciences](https://www.mdpi.com/2076-3417/15/13/6958)
- Shankar et al. (2022) — Augmentation effects in SER: speed perturbation is the strongest single augmentation · [arXiv:2211.05047](https://arxiv.org/abs/2211.05047)
- Ye et al. (2023) — TIM-Net: multi-scale temporal modeling, 92.08% UAR LOSO on RAVDESS · [ICASSP 2023](https://github.com/Jiaxin-Ye/TIM-Net_SER)
- Lin et al. (2017) — Focal Loss for Dense Object Detection · [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- Busso et al. (2013) — Iterative Feature Normalization for emotional speech · [IEEE TAC](https://lab-msp.com/CarlosBusso/publications/Busso_2013_2.pdf)
- Hochreiter & Schmidhuber (1997) — Long Short-Term Memory
- Cho et al. (2014) — Learning Phrase Representations using RNN Encoder-Decoder
- Fayek H (2016) — Speech Processing for Machine Learning · [haythamfayek.com](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
- Practical Cryptography — Guide to Mel Frequency Cepstral Coefficients · [practicalcryptography.com](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)
