from flask import Flask, request, render_template, redirect, url_for, session, abort
from detection_utils import process_media
import os
import uuid
import json

# Setup Flask app
app = Flask(__name__)
# IMPORTANT: A secret key is required to use sessions
app.secret_key = str(uuid.uuid4())

@app.route('/')
def home():
    """Renders the main input form (index.html)."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """
    Processes the media file path, calls the updated process_media function,
    and stores the dictionary result in the session.
    """
    file_path = request.form.get('file_path')
    if not file_path:
        return "No file path provided.", 400

    # Call the processing function, which returns a dictionary {output_path, detections, ...}
    try:
        results = process_media(file_path)
    except Exception as e:
        print(f"Processing error: {e}")
        # Return a user-friendly error message
        return f"Error processing media. Details: {e}", 500

    if not results or 'output_path' not in results:
         return "Processing failed to return valid results.", 500

    # Generate a unique ID for this result set and store the entire results dictionary in the session
    result_id = str(uuid.uuid4())
    session[result_id] = results
    
    # Redirect to the display route using the unique result ID
    return redirect(url_for('display_output', result_id=result_id))

@app.route('/display/<result_id>')
def display_output(result_id):
    """
    Displays the structured output by retrieving data from the session.
    """
    # Retrieve the results dictionary from the session
    results = session.get(result_id)

    if not results:
        return "Result not found or session expired. Please process the media again.", 404

    # The output path is now nested inside the results dictionary
    output_filename = os.path.basename(results['output_path'])
    
    # Determine if the file is a video (for display purposes in the template)
    is_video = results['file_path'].endswith(('.mp4', '.avi', '.mov', '.mkv'))

    # Pass the output filename using BOTH the new and old variable names ('filename' and 'output_filename')
    # to satisfy older templates that expect 'filename'.
    return render_template(
        'output.html', 
        output_filename=output_filename,
        filename=output_filename, # Added for backward compatibility with your template
        results=results, 
        is_video=is_video
    )

if __name__ == '__main__':
    app.run(debug=True)
