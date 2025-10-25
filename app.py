# app.py

from flask import Flask, request, render_template, redirect, url_for
from detection_utils import process_media
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    file_path = request.form.get('file_path')
    if not file_path:
        return "No file path provided.", 400

    # This is a synchronous call. The page will hang here.
    output_path = process_media(file_path)

    if output_path:
        return redirect(url_for('display_output', filename=os.path.basename(output_path)))
    else:
        return "Error processing media. Check the file path.", 500

@app.route('/display/<filename>')
def display_output(filename):
    is_video = filename.endswith(('.mp4', '.avi', '.mov', '.mkv'))
    return render_template('output.html', filename=filename, is_video=is_video)

if __name__ == '__main__':
    app.run(debug=True)