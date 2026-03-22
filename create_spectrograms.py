import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

dataset_path = "dataset"
output_path = "spectrograms"

labels = ["hungry", "sleepy", "discomfort", "gas"]

for label in labels:
    os.makedirs(f"{output_path}/{label}", exist_ok=True)
    
    for file in os.listdir(f"{dataset_path}/{label}"):
        file_path = f"{dataset_path}/{label}/{file}"
        
        try:
            audio, sr = librosa.load(file_path)
            
            plt.figure(figsize=(3,3))
            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
            
            librosa.display.specshow(spectrogram_db, sr=sr)
            plt.axis('off')
            
            save_path = f"{output_path}/{label}/{file}.png"
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        except:
            print("Error:", file_path)

print("✅ Spectrograms created!")