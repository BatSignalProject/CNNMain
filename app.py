import os
from flask import Flask, render_template, request, send_from_directory
import librosa
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt

# Initialize the Flask app and set configurations for upload and spectrogram directories
app = Flask(__name__)

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads' 
SPECTROGRAM_FOLDER = 'spectrograms' # Folder to store generated spectrogram images
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SPECTROGRAM_FOLDER'] = SPECTROGRAM_FOLDER

# Define a fixed size for spectrogram images to standardize output dimensions
fixed_size = (640, 640)

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
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename) # Define upload path

        # Save the uploaded file to the server
        file.save(file_path) 

        # Load the audio file using librosa
        y, sr = librosa.load(file_path)
        
        # Apply a band-pass filter to the audio to isolate the desired frequency range
        low_freq = 300  # Lower bound of the frequency range (in Hz)
        high_freq = 3000  # Upper bound of the frequency range (in Hz)

        # Nyquist frequency for normalization
        nyquist = 0.5 * sr
        low = low_freq / nyquist
        high = high_freq / nyquist

        # Design a Butterworth band-pass filter
        b, a = scipy.signal.butter(1, [low, high], btype='band')

        # Apply the filter to the audio signal
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

        # Set figure size for the plot
        plt.figure(figsize=(10, 10))

        # Display the spectrogram
        plt.imshow(spectrogram, aspect='auto', origin='lower')

        # Define save path
        spectrogram_image_path = os.path.join(app.config['SPECTROGRAM_FOLDER'], 'spectrogram.png')

        # Save the plot as an image file
        plt.savefig(spectrogram_image_path)
        plt.close()

        # Render the results page with the generated spectrogram image
        return render_template('results.html', image_path='spectrograms/spectrogram.png')
    else:
        return 'Invalid file format'
    
# Define a route to serve the generated spectrogram image
@app.route('/spectrograms/<filename>')
def send_spectrogram(filename):
    return send_from_directory(app.config['SPECTROGRAM_FOLDER'], filename)

# Run the app in debug mode for easier development and debugging
if __name__ == "__main__":
    app.run(debug=True)