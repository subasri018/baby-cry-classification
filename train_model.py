import os
import librosa
import numpy as np
from sklearn.svm import SVC
import joblib

X = []
y = []

labels = ["hungry", "sleepy", "discomfort", "gas"]

for label in labels:
    folder = f"dataset/{label}"
    
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        
        try:
            audio, sr = librosa.load(file_path)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            mfcc_mean = np.mean(mfcc.T, axis=0)
            
            X.append(mfcc_mean)
            y.append(label)
        
        except Exception as e:
            print("Error in file:", file_path)

# Train model
model = SVC(probability=True)
model.fit(X, y)

# Save model
joblib.dump(model, "baby_model.pkl")

print("✅ Model trained and saved!")