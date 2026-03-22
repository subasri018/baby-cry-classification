import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

# Load model
model = tf.keras.models.load_model("cnn_model.h5")

labels = ["discomfort", "gas", "hungry", "sleepy"]

fs = 44100
seconds = 3

print("🎤 Recording...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

write("test.wav", fs, recording)

# Create spectrogram
audio, sr = librosa.load("test.wav")

plt.figure(figsize=(3,3))
spec = librosa.feature.melspectrogram(y=audio, sr=sr)
spec_db = librosa.power_to_db(spec, ref=np.max)

librosa.display.specshow(spec_db, sr=sr)
plt.axis('off')
plt.savefig("test.png", bbox_inches='tight', pad_inches=0)
plt.close()

# Load image
img = cv2.imread("test.png")
img = cv2.resize(img, (128,128))
img = img / 255.0
img = np.reshape(img, (1,128,128,3))

# Predict
prediction = model.predict(img)
class_index = np.argmax(prediction)
confidence = np.max(prediction)

print("👶 Baby needs:", labels[class_index])
print("Confidence:", confidence * 100, "%")