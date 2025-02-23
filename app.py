import os
from flask import Flask, render_template, request, send_from_directory
import librosa
import matplotlib.pyplot as plt
import numpy as np

# Initialize the Flask app and set configurations for upload and spectrogram directories
app = Flask(__name__)

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads' 
SPECTROGRAM_FOLDER = 'spectrograms' # Folder to store generated spectrogram images
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SPECTROGRAM_FOLDER'] = SPECTROGRAM_FOLDER

# Define the root route that renders the upload form HTML page
@app.route('/')
def upload_form():
    return render_template('upload.html') # Render the upload form (upload.html)

# Define the route to handle file uploads and process them
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file is included in the request
    if 'file' not in request.files:
        return 'No file part'
    
    # Get the file from the request
    file = request.files['file'] 

    # Check if the file name is empty
    if file.filename == '': 
        return 'No selected file'
    
    # Validate that the file is a .wav file
    if file and file.filename.endswith('.wav'):
        filename = file.filename

        # Define upload path
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save the uploaded file to the server
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

        # Render the results page with the generated spectrogram
        return render_template('results.html', image_path='spectrograms/spectrogram.png')
    else:
        return 'Invalid file format'
    
@app.route('/spectrograms/<filename>')
def send_spectrogram(filename):
    return send_from_directory(app.config['SPECTROGRAM_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)