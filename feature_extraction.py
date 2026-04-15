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

    # processing 

    # Pre emphasis
    alpha = 0.97
    peAudio = preEmphasisFilt(audio,alpha=alpha)

    # Features
    # MFCC
    mfcc = librosa.feature.mfcc(y=peAudio,sr=sr,n_mfcc=40,n_fft = 2048, win_length = win_len, hop_length = win_hop)

    # Deltas
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order = 2)

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y=peAudio,frame_length = win_len, hop_length = win_hop)

    # RMS
    rms = librosa.feature.rms(y=peAudio,frame_length = win_len,hop_length = win_hop)
    
    # Put features into dict for easy loading
    feat_dict = {
        "actor_id": actor_id,       # Actor ID
        "gender": actor_id % 2,     # Gender (1: male, 0: female)
        "emotion": emotion_code,    # Numerized emotion (see EMOTION_MAP)
        "mfcc": mfcc,               # MFCC matrix [40, Window number]
        "delta": delta,             # delta matrix [40, Window number]
        "delta2": delta2,           # delta-delta matrix [40, Window number]
        "zcr": zcr,                 # ZCR vector [1, Window number]
        "rms": rms                  # RMS vector [1, Window number]
    }

    output_path = filepath.replace('.wav', '.npz')
    np.savez(output_path, **feat_dict)




