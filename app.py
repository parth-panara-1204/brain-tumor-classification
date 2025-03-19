from flask import Flask, render_template, request, redirect, url_for, session
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("brain_tumor_model.keras")

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
    if 'mri-file' not in request.files:
        return redirect(url_for('upload'))
    
    file = request.files['mri-file']
    if file.filename == '':
        return redirect(url_for('upload'))
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    image = Image.open(filepath)
    image = image.resize((150,150))
    image_array = np.array(image)

    if image_array.ndim == 2:  # If it's grayscale
        image_array = np.stack((image_array,) * 3, axis=-1)

    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0

    prediction = model.predict(image_array)
    pred_values = prediction[0].tolist()

    class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    highest_prob_index = np.argmax(pred_values)
    diagnosis = class_names[highest_prob_index]
    confidence = pred_values[highest_prob_index]
    
    # Store results in session
    session['result'] = {
        'prediction_values': pred_values,
        'class_names': class_names,
        'diagnosis': diagnosis,
        'confidence': confidence,
        'image_path': filepath
    }

    session['result']['predictions'] = list(zip(class_names, pred_values))
    
    return redirect(url_for('results'))

@app.route('/results')
def results():
    return render_template('results.html', result=session.get('result'))

if __name__ == '__main__':
    app.run(debug=True)