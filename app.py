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
        
        # Apply band-pass filter
        low_freq = 300  # Lower bound of the frequency range
        high_freq = 3000  # Upper bound of the frequency range
        nyquist = 0.5 * sr
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = scipy.signal.butter(1, [low, high], btype='band')
        y_filtered = scipy.signal.lfilter(b, a, y)

        # Detect and remove silent parts
        non_silent_intervals = librosa.effects.split(y_filtered, top_db=20)
        y_non_silent = np.concatenate([y_filtered[start:end] for start, end in non_silent_intervals])

        # Convert to spectrogram
        spectrogram = librosa.feature.melspectrogram(y=y_non_silent, sr=sr)

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