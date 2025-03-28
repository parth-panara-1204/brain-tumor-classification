from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import uuid  # For generating unique filenames

# Load the model
model = load_model("brain_tumor_model.keras")

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.secret_key = 'asdf23fswef3awefs'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Check if a file is uploaded
    if 'mri-file' not in request.files:
        flash("No file part in the request.")
        return redirect(url_for('upload'))
    
    file = request.files['mri-file']
    if file.filename == '':
        flash("No file selected.")
        return redirect(url_for('upload'))
    
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        flash("Unsupported file type. Please upload a JPG, JPEG, or PNG file.")
        return redirect(url_for('upload'))
    
    # Save the file with a unique name
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    # Preprocess the image
    try:
        image = Image.open(filepath)
        image = image.resize((150, 150))
        image_array = np.array(image)

        # Handle grayscale images
        if image_array.ndim == 2:  # If it's grayscale
            image_array = np.stack((image_array,) * 3, axis=-1)

        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0
    except Exception as e:
        flash(f"Error processing the image: {str(e)}")
        return redirect(url_for('upload'))

    # Make predictions
    prediction = model.predict(image_array)
    pred_values = prediction[0].tolist()

    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    highest_prob_index = np.argmax(pred_values)
    diagnosis = class_names[highest_prob_index]
    confidence = pred_values[highest_prob_index]
    
    # Store results in session
    session['result'] = {
        'image_path': f'uploads/{unique_filename}',  # Relative path for the static folder
        'diagnosis': diagnosis,
        'confidence': confidence,
        'predictions': list(zip(class_names, pred_values))
    }

    session['result']['predictions'] = list(zip(class_names, pred_values))
    
    return redirect(url_for('results'))

@app.route('/results')
def results():
    result = session.get('result')
    if not result:
        flash("No results available. Please upload an MRI scan.")
        return redirect(url_for('upload'))
    return render_template('results.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)