import os
import librosa
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
import collections

# List to hold spectrograms and labels
spectrograms = []
labels = []

# Define a fixed size for your spectrograms
fixed_size = (128, 128)

# Create a directory to save spectrograms (FOR TESTING, REMOVE LATER)
# spectrogram_dir = 'spectrograms_training'
# os.makedirs(spectrogram_dir, exist_ok=True)

# Loop over all directories in the parent directory
for species_dir in ['BatNYCLEI', 'BatPIPPIP', 'BatMYOSPP2']:
    wav_subdir = os.path.join('batdata', species_dir)
    for filename in os.listdir(wav_subdir):
        if filename.endswith('.wav'):
            # Load .wav file
            y, sr = librosa.load(os.path.join(wav_subdir, filename), sr=None)
        
            # Find the loudest point in the audio signal
            loudest_point = np.argmax(np.abs(y))

            # Calculate the sample index for 0.02 seconds before and after the loudest point
            trim_duration = int(sr * 0.02)
            start_point = max(0, loudest_point - trim_duration)
            end_point = min(len(y), loudest_point + trim_duration)

            # Trim the audio signal
            y = y[start_point:end_point]
        
            # Convert to spectrogram
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, fmin=20000, fmax=80000, n_fft=1024, hop_length=256)

            # Save spectrogram (FOR TESTING, REMOVE LATER)
            # plt.figure(figsize=(4, 4))
            # librosa.display.specshow(librosa.power_to_db(spectrogram), sr=sr, y_axis='mel', fmin=20000, fmax=80000, cmap="gray_r", vmin=-60, vmax=20)
            # plt.ylabel('Frequency (Hz)')
            # plt.yticks([20000, 30000, 40000, 50000, 60000, 70000, 80000])
            # spectrogram_image_path = os.path.join(spectrogram_dir, f"{species_dir}_{filename.replace('.wav', '.png')}")
            # plt.savefig(spectrogram_image_path)
            # plt.close()
            
            # Resize spectrogram to fixed size using interpolation
            zoom_factor = (fixed_size[0] / spectrogram.shape[0], fixed_size[1] / spectrogram.shape[1])
            spectrogram = scipy.ndimage.zoom(spectrogram, zoom_factor)

            spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))

            # Flatten spectrogram and add to list
            spectrograms.append(spectrogram.flatten())

            # Add species to labels list
            labels.append(species_dir)

            print(f'{filename} converted to spectrogram.')

# Convert list to numpy array
spectrograms = np.array(spectrograms)

# Convert labels to integers
label_dict = {label: i for i, label in enumerate(set(labels))}
labels = [label_dict[label] for label in labels]


print(collections.Counter(labels))

# Save the label dictionary to a JSON file
with open('label_dict.json', 'w') as f:
    json.dump(label_dict, f)

print("Label dictionary saved to label_dict.json")

labels = to_categorical(labels)

X_train, X_temp, y_train, y_temp = train_test_split(spectrograms, labels, test_size=0.3, random_state=42, stratify=labels)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

X_train = X_train + np.random.normal(0, 0.01, X_train.shape)

# Save the testing data
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

# Reshape data to fit the model input
X_train = X_train.reshape(-1, fixed_size[0], fixed_size[1], 1)
X_val = X_val.reshape(-1, fixed_size[0], fixed_size[1], 1)
X_test = X_test.reshape(-1, fixed_size[0], fixed_size[1], 1)


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(fixed_size[0], fixed_size[1], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# Define the checkpoint callback
# checkpoint = ModelCheckpoint(filepath='best_model3.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=16, callbacks=[early_stopping, lr_schedule])

model.save('model.h5')