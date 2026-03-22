import cv2
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
import joblib
import threading

model = joblib.load("baby_model.pkl")

fs = 44100
seconds = 3

prediction_text = "Press 'r' to detect"

def record_and_predict():
    global prediction_text

    print("🎤 Recording...")

    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write("test.wav", fs, recording)

    audio, sr = librosa.load("test.wav")
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

    prediction = model.predict(mfcc_mean)[0]

    prediction_text = f"Baby needs: {prediction}"
    print(prediction_text)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Show prediction text continuously
    cv2.putText(frame, prediction_text,
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Baby Monitor", frame)

    key = cv2.waitKey(1)

    if key == ord('r'):
        # Run recording in separate thread
        threading.Thread(target=record_and_predict).start()

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()