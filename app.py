import os
from flask import Flask, render_template, request, send_from_directory, send_file
from keras.models import load_model
import librosa
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import json
import zipfile
import tempfile
import csv

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads' 
SPECTROGRAM_FOLDER = 'spectrograms'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SPECTROGRAM_FOLDER'] = SPECTROGRAM_FOLDER

# Load the model
model = load_model('model6.h5')

# Define a fixed size for your spectrograms
fixed_size = (640, 640)

# Load the label dictionary from the JSON file
with open('label_dict.json', 'r') as f:
    label_dict = json.load(f)

# Reverse the dictionary to map indices to labels
label_dict = {v: k for k, v in label_dict.items()}

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and file.filename.endswith('.wav'):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load the audio file using librosa
        y, sr = librosa.load(file_path, sr=None)
        
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

        # Save spectrogram
        plt.figure(figsize=(4, 4))
        librosa.display.specshow(librosa.power_to_db(spectrogram), sr=sr, y_axis='mel', fmin=20000, fmax=80000, cmap="gray_r", vmin=-60, vmax=20)
        plt.ylabel('Frequency (Hz)')
        plt.yticks([20000, 30000, 40000, 50000, 60000, 70000, 80000])
        spectrogram_image_path = os.path.join(app.config['SPECTROGRAM_FOLDER'], 'spectrogram.png')
        plt.savefig(spectrogram_image_path)
        plt.close()

        # Resize spectrogram to fixed size using interpolation
        zoom_factor = (fixed_size[0] / spectrogram.shape[0], fixed_size[1] / spectrogram.shape[1])
        spectrogram = scipy.ndimage.zoom(spectrogram, zoom_factor)

        # Flatten spectrogram
        spectrogram = spectrogram.flatten()

        # Normalize spectrogram
        spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))

        # Reshape spectrogram to include channel dimension
        spectrogram = spectrogram.reshape(1, fixed_size[0], fixed_size[1], 1)

        # Make prediction
        y_pred = model.predict(spectrogram)

        # Map predictions to labels dynamically
        predictions = {label_dict[i]: y_pred[0][i] * 100 for i in range(len(y_pred[0]))}

        # Extract confidence values
        confidence_nyclei = predictions['BatNYCLEI']
        confidence_pippip = predictions['BatPIPPIP']
        confidence_myospp = predictions['BatMYOSPP2']

        return render_template('results.html', confidence_nyclei=confidence_nyclei, confidence_pippip=confidence_pippip, confidence_myospp=confidence_myospp, image_path='spectrograms/spectrogram.png')
    else:
        return 'Invalid file format'

@app.route('/upload_folder', methods=['POST'])
def upload_folder():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and file.filename.endswith('.zip'):
        # Save the uploaded .zip file to a temporary location
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, file.filename)
        file.save(zip_path)

        # Extract the .zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Process each .wav file in the extracted folder
        results = []
        for root, _, files in os.walk(temp_dir):
            for filename in files:
                if filename.endswith('.wav'):
                    file_path = os.path.join(root, filename)

                    # Process the .wav file
                    y, sr = librosa.load(file_path, sr=None)
                    loudest_point = np.argmax(np.abs(y))
                    trim_duration = int(sr * 0.02)
                    start_point = max(0, loudest_point - trim_duration)
                    end_point = min(len(y), loudest_point + trim_duration)
                    y = y[start_point:end_point]
                    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, fmin=20000, fmax=80000, n_fft=1024, hop_length=256)
                    zoom_factor = (fixed_size[0] / spectrogram.shape[0], fixed_size[1] / spectrogram.shape[1])
                    spectrogram = scipy.ndimage.zoom(spectrogram, zoom_factor)
                    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
                    spectrogram = spectrogram.reshape(1, fixed_size[0], fixed_size[1], 1)

                    # Make prediction
                    y_pred = model.predict(spectrogram)
                    predictions = {label_dict[i]: y_pred[0][i] * 100 for i in range(len(y_pred[0]))}

                    # Append results
                    results.append({
                        'file': filename,
                        'confidence_nyclei': predictions['BatNYCLEI'],
                        'confidence_pippip': predictions['BatPIPPIP'],
                        'confidence_myospp': predictions['BatMYOSPP2']
                    })

        # Generate a CSV file
        csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.csv')
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['file', 'confidence_nyclei', 'confidence_pippip', 'confidence_myospp'])
            writer.writeheader()
            writer.writerows(results)

        csv_file_name = os.path.basename(csv_file_path)

        # Render results
        return render_template('results_folder.html', results=results, csv_file_name=csv_file_name)
    else:
        return 'Invalid file format'
    
@app.route('/spectrograms/<filename>')
def send_spectrogram(filename):
    return send_from_directory(app.config['SPECTROGRAM_FOLDER'], filename)

@app.route('/download_csv/<filename>')
def download_csv(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True, download_name='predictions.csv')

if __name__ == "__main__":
    app.run(debug=True)