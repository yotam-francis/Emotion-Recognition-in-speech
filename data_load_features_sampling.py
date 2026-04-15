import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Load Sample Audio
filename_neutral = "Audio_Speech_Actors_01-24\\Actor_01\\03-01-01-01-01-01-01.wav" # Neutral emotion
audio_neutral, sr_net = librosa.load(filename_neutral, sr = None)

filename_angry = "Audio_Speech_Actors_01-24\\Actor_01\\03-01-05-01-01-01-01.wav" # Angry emotion
audio_angry,sr_angry = librosa.load(filename_angry,sr = None)

# Plot in time domain
plt.figure()
plt.subplot(2,1,1)
librosa.display.waveshow(audio_neutral, sr=sr_net)
plt.title("Actor 01 - Audio - Speech - Neutral - Normal - Kids - 1st rep")
plt.xlabel('Time[s]')
plt.ylabel('Amplitude')

plt.subplot(2,1,2)
librosa.display.waveshow(audio_angry, sr=sr_angry)
plt.title("Actor 01 - Audio - Speech - Angry - Normal - Kids - 1st rep")
plt.xlabel('Time[s]')
plt.ylabel('Amplitude')

plt.show()

# Pre - Emphasis
def preEmphasisFilt(sig,alpha):
    b = [1, -alpha]
    filtred_signal = signal.lfilter(b,1,sig)
    return filtred_signal

alpha = 0.97
ephd_neutral = preEmphasisFilt(audio_neutral,alpha)
ephd_angry = preEmphasisFilt(audio_angry,alpha)

# Plot in time domain
plt.figure()
plt.subplot(2,1,1)
librosa.display.waveshow(ephd_neutral, sr=sr_net)
plt.title("Actor 01- Neutral - Normal - Kids - 1st rep - pre emphasiezed")
plt.xlabel('Time[s]')
plt.ylabel('Amplitude')

plt.subplot(2,1,2)
librosa.display.waveshow(ephd_angry, sr=sr_angry)
plt.title("Actor 01 - Audio - Speech - Angry - Normal - Kids - 1st rep - pre emphasiezed")
plt.xlabel('Time[s]')
plt.ylabel('Amplitude')

plt.show()

# Feature extraction

# Frame defintion for MFCC, STFT, ZCR etc.

# Speech is non stationary - we assume that in short time frames (20-40ms) it is stationary
# 15ms overlap 60% overlap
frame_time = 0.025 # 25ms
frame_stride = 0.01 # 10ms

# Seconds to samples
frame_length = int(frame_time*sr_net) # sample rate is the same for all audio files here (48kHz)
frame_step = int(frame_stride*sr_net)

# MFCC - Mel-frequency cepstral coefficents
# Explained here :(https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)

coeff_num = 40 # Number of coefficents for MFCC
mfcc_neutral = librosa.feature.mfcc(y=ephd_neutral,
                                sr=sr_net, 
                                n_mfcc = coeff_num,
                                n_fft = 2048, # FFT works best at powers of 2
                                win_length = frame_length, # win_len < n_fft therefore for a 1200 frame length 2048 fft was chosen
                                hop_length = frame_step)

print(mfcc_neutral.shape) # mfcc_neutral will be of shape (coefficents,windows)

# Repeat for angry recording
mfcc_angry = librosa.feature.mfcc(y=ephd_angry,
                                sr=sr_net,
                                n_mfcc = coeff_num,
                                n_fft = 2048, 
                                win_length = frame_length,
                                hop_length = frame_step)

print(mfcc_angry.shape)

# Mel Spectrograms
mel_neutral = librosa.feature.melspectrogram(y=ephd_neutral,
                                              sr=sr_net,
                                              n_fft=2048,
                                              win_length=frame_length,
                                              hop_length=frame_step)

mel_angry = librosa.feature.melspectrogram(y=ephd_angry,
                                            sr=sr_net,
                                            n_fft=2048,
                                            win_length=frame_length,
                                            hop_length=frame_step)

# Convert to dB scale for better visualization
mel_neutral_db = librosa.power_to_db(mel_neutral, ref=np.max)
mel_angry_db = librosa.power_to_db(mel_angry, ref=np.max)

# Visualise and compare MFCCs and mel-spectrograms
# MFCC
plt.figure()
plt.subplot(2,1,1)
librosa.display.specshow(mfcc_neutral, sr=sr_net, hop_length=frame_step, x_axis='time')
plt.colorbar()
plt.title("MFCC - Neutral")
plt.xlabel('Time[s]')
plt.ylabel('MFCC Coefficients')

plt.subplot(2,1,2)
librosa.display.specshow(mfcc_angry, sr=sr_net, hop_length=frame_step, x_axis='time')
plt.colorbar()
plt.title("MFCC - Angry")
plt.xlabel('Time[s]')
plt.ylabel('MFCC Coefficients')

plt.tight_layout()
plt.show()

# Mel-spectrograms
plt.figure()
plt.subplot(2,1,1)
librosa.display.specshow(mel_neutral_db, sr=sr_net, hop_length=frame_step, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram - Neutral")
plt.xlabel('Time[s]')
plt.ylabel('Frequency [Hz]')

plt.subplot(2,1,2)
librosa.display.specshow(mel_angry_db, sr=sr_net, hop_length=frame_step, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram - Angry")
plt.xlabel('Time[s]')
plt.ylabel('Frequency [Hz]')

plt.tight_layout()
plt.show()

# Delta and delta-delta
# First and second order derivative of the mfcc coeffs 
# These represnt the porsodic movements of speech (rhythm, stress, intonation) and the dynamics of thos accordingly

delta_neutral = librosa.feature.delta(mfcc_neutral)
delta_delta_neutral = librosa.feature.delta(mfcc_neutral,order = 2)
print(f"delta neutral shape is:{delta_neutral.shape}.\ndelta delta of nuetral is in shape of {delta_delta_neutral.shape}")

delta_angry = librosa.feature.delta(mfcc_angry)
delta_delta_angry = librosa.feature.delta(mfcc_angry,order = 2)
print(f"delta angry shape is:{delta_angry.shape}.\ndelta delta of angry is in shape of {delta_delta_angry.shape}")

# Plot Deltas
plt.figure()
plt.subplot(2,1,1)
librosa.display.specshow(delta_neutral, sr=sr_net, hop_length=frame_step, x_axis='time')
plt.colorbar()
plt.title("Delta MFCC - Neutral")
plt.xlabel('Time[s]')
plt.ylabel('MFCC Coefficients')

plt.subplot(2,1,2)
librosa.display.specshow(delta_angry, sr=sr_net, hop_length=frame_step, x_axis='time')
plt.colorbar()
plt.title("Delta MFCC - Angry")
plt.xlabel('Time[s]')
plt.ylabel('MFCC Coefficients')

plt.tight_layout()
plt.show()

# Plot Delta-Deltas
plt.figure()
plt.subplot(2,1,1)
librosa.display.specshow(delta_delta_neutral, sr=sr_net, hop_length=frame_step, x_axis='time')
plt.colorbar()
plt.title("Delta-Delta MFCC - Neutral")
plt.xlabel('Time[s]')
plt.ylabel('MFCC Coefficients')

plt.subplot(2,1,2)
librosa.display.specshow(delta_delta_angry, sr=sr_net, hop_length=frame_step, x_axis='time')
plt.colorbar()
plt.title("Delta-Delta MFCC - Angry")
plt.xlabel('Time[s]')
plt.ylabel('MFCC Coefficients')

plt.tight_layout()
plt.show()

# ZCR(Zeo crossing rate)
# How many times the signal crosses the zero amplitude in a given window (indicative of voiced vs. non voices fonems)
zcr_neutral = librosa.feature.zero_crossing_rate(ephd_neutral,frame_length=frame_length,hop_length = frame_step)
zcr_angry = librosa.feature.zero_crossing_rate(ephd_angry,frame_length=frame_length,hop_length = frame_step)

# Plot ZCR
plt.figure()
plt.subplot(2,1,1)
time_axis_neutral = librosa.frames_to_time(np.arange(zcr_neutral.shape[1]), sr=sr_net, hop_length=frame_step)
plt.plot(time_axis_neutral, zcr_neutral[0])
plt.title("ZCR - Neutral")
plt.xlabel('Time[s]')
plt.ylabel('Zero Crossing Rate')

plt.subplot(2,1,2)
time_axis_angry = librosa.frames_to_time(np.arange(zcr_angry.shape[1]), sr=sr_net, hop_length=frame_step)
plt.plot(time_axis_angry, zcr_angry[0])
plt.title("ZCR - Angry")
plt.xlabel('Time[s]')
plt.ylabel('Zero Crossing Rate')

plt.tight_layout()
plt.show()

# RMS (Root Mean Square)
# The energy of a given window
rms_neutral = librosa.feature.rms(y=ephd_neutral, frame_length=frame_length, hop_length=frame_step)
rms_angry = librosa.feature.rms(y=ephd_angry, frame_length=frame_length, hop_length=frame_step)

# Plot RMS
plt.figure()
plt.subplot(2,1,1)
time_axis_neutral = librosa.frames_to_time(np.arange(rms_neutral.shape[1]), sr=sr_net, hop_length=frame_step)
plt.plot(time_axis_neutral, rms_neutral[0])
plt.title("RMS Energy - Neutral")
plt.xlabel('Time[s]')
plt.ylabel('RMS Energy')

plt.subplot(2,1,2)
time_axis_angry = librosa.frames_to_time(np.arange(rms_angry.shape[1]), sr=sr_net, hop_length=frame_step)
plt.plot(time_axis_angry, rms_angry[0])
plt.title("RMS Energy - Angry")
plt.xlabel('Time[s]')
plt.ylabel('RMS Energy')

plt.tight_layout()
plt.show()