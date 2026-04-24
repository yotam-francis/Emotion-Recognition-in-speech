import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import glob
import os
from tqdm import tqdm

# Here We will load each audio file and extract the features with the relevant tags
# This will be done individually to not take up a lot of memory

# RAVDESS filename encoding:
# Modality - VocalChannel - Emotion - Intensity - Statement - Repetition - Actor
# Emotion codes:

EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

dataset_path = "Audio_Speech_Actors_01-24"
audio_files = glob.glob(os.path.join(dataset_path, "Actor_*", "*.wav"))
win_len = int(48000*0.025) # 25ms window for sampling rate 48kHz
win_hop = int(0.015*48000) # 15ms stride for sr 48kHz (60% overlap)

def augment_audio(audio, sr):
    augmented = []
    
    # pitch shift up by 2 semitones
    augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=2))
    # pitch shift down by 2 semitones
    augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2))
    # time stretch faster
    augmented.append(librosa.effects.time_stretch(audio, rate=1.1))
    # time stretch slower
    augmented.append(librosa.effects.time_stretch(audio, rate=0.9))
    
    return augmented

def preEmphasisFilt(sig,alpha):
    b = [1, -alpha]
    filtred_signal = signal.lfilter(b,1,sig)
    return filtred_signal


for filepath in tqdm(audio_files, desc="Extracting features"):
    filename = os.path.basename(filepath)
    parts = filename.replace('.wav', '').split('-')

    emotion_code = parts[2]
    emotion = EMOTION_MAP[emotion_code]
    actor_id = int(parts[6])
    gender = 'male' if actor_id % 2 == 1 else 'female'
    
    # Load audio
    audio, sr = librosa.load(filepath, sr=None)

    all_versions = [audio] + augment_audio(audio, sr)
    alpha = 0.97

    def norm(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    for i, sig in enumerate(all_versions):
        peAudio = preEmphasisFilt(sig, alpha=alpha)

        mfcc = librosa.feature.mfcc(y=peAudio, sr=sr, n_mfcc=40, n_fft=2048, win_length=win_len, hop_length=win_hop)
        mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        zcr = librosa.feature.zero_crossing_rate(y=peAudio, frame_length=win_len, hop_length=win_hop)
        rms = librosa.feature.rms(y=peAudio, frame_length=win_len, hop_length=win_hop)

        mel = librosa.feature.melspectrogram(y=peAudio, sr=sr, n_mels=64, hop_length=win_hop, win_length=win_len)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        delta_mel = librosa.feature.delta(mel_db)
        delta2_mel = librosa.feature.delta(mel_db, order=2)
        mel_3ch = np.stack([norm(mel_db), norm(delta_mel), norm(delta2_mel)], axis=0)  # (3, 64, T)

        zcr = norm(zcr)
        rms = norm(rms)

        feat_dict = {
            "actor_id": actor_id,
            "gender": actor_id % 2,
            "emotion": emotion_code,
            "mfcc": mfcc,
            "delta": delta,
            "delta2": delta2,
            "zcr": zcr,
            "rms": rms,
            "mel3ch": mel_3ch,
        }

        suffix = f'-{i}'  # -0 original, -1 to -4 augmented
        output_path = filepath.replace('.wav', f'{suffix}.npz')
        np.savez(output_path, **feat_dict)



