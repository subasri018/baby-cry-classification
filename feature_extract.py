import librosa
import numpy as np

audio, sr = librosa.load("output.wav")

mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

print("MFCC Shape:", mfcc.shape)