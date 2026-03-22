import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import joblib

# Load trained model
model = joblib.load("baby_model.pkl")

fs = 44100
seconds = 3

print("🎤 Recording...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

write("test.wav", fs, recording)

# Load audio
audio, sr = librosa.load("test.wav")

# Extract MFCC
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)

# Predict
prediction = model.predict(mfcc_mean)

print("👶 Baby needs:", prediction[0])