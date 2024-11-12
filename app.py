import os
from flask import Flask, render_template, request, send_from_directory
import librosa
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
SPECTROGRAM_FOLDER = 'spectrograms'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SPECTROGRAM_FOLDER'] = SPECTROGRAM_FOLDER

# Define a fixed size for your spectrograms
fixed_size = (640, 640)

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

        # Load .wav file
        y, sr = librosa.load(file_path)

        # Convert to spectrogram
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

        # Resize spectrogram to fixed size using interpolation
        zoom_factor = (fixed_size[0] / spectrogram.shape[0], fixed_size[1] / spectrogram.shape[1])
        spectrogram = scipy.ndimage.zoom(spectrogram, zoom_factor)
        
        # Save spectrogram as an image file
        plt.figure(figsize=(10, 10))
        plt.imshow(spectrogram, aspect='auto', origin='lower')
        spectrogram_image_path = os.path.join(app.config['SPECTROGRAM_FOLDER'], 'spectrogram.png')
        plt.savefig(spectrogram_image_path)
        plt.close()

        return render_template('results.html', image_path='spectrograms/spectrogram.png')
    else:
        return 'Invalid file format'
    
@app.route('/spectrograms/<filename>')
def send_spectrogram(filename):
    return send_from_directory(app.config['SPECTROGRAM_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)