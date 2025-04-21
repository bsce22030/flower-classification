import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

# Set up folders for image uploads and processed results
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['PROCESSED_FOLDER'] = './static/processed'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Load your trained Keras model (ensure the model is in the same directory or provide the correct path)
model = load_model('flower_classifier.h5')  # Change this to the correct model file path

# Class names for the Oxford 102 Flower Dataset (adjust this as needed)
class_names = [
    'Aconitum', 'Actaea', 'Agapanthus', 'Alchemilla', 'Anemone', 'Arctostaphylos', 'Armeria', 'Asclepias',
    'Astilbe', 'Aubrieta', 'Azalea', 'Bergenia', 'Calibrachoa', 'Calluna', 'Campanula', 'Cannabis',
    'Clematis', 'Columbine', 'Coreopsis', 'Corydalis', 'Crocus', 'Dahlia', 'Delphinium', 'Dicentra', 'Echinacea',
    'Episcia', 'Erica', 'Eryngium', 'Freesia', 'Fuchsia', 'Galax', 'Gardenia', 'Geranium', 'Gloxinia', 'Helleborus',
    'Hemerocallis', 'Hibiscus', 'Hosta', 'Impatiens', 'Ipomoea', 'Iris', 'Kalanchoe', 'Kniphofia', 'Lobelia',
    'Lupinus', 'Mandevilla', 'Monarda', 'Nicotiana', 'Oenothera', 'Osteospermum', 'Papaver', 'Passiflora', 'Pelargonium',
    'Penstemon', 'Petunia', 'Phlox', 'Plumbago', 'Potentilla', 'Primula', 'Rhododendron', 'Rudbeckia', 'Salvia', 'Scabiosa',
    'Sedges', 'Senecio', 'Silene', 'Sinningia', 'Soleirolia', 'Solanum', 'Stachys', 'Sundew', 'Sutera', 'Tropaeolum', 
    'Tulip', 'Verbena', 'Vinca', 'Viola', 'Vitis', 'Zinnia'
]  # Replace with actual class names from your dataset

# Define image size expected by the model
img_size = (224, 224)  # MobileNetV2 standard size (resize to this size for prediction)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            return "No file selected", 400
        
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the uploaded image
            result_image_url, flower_name = process_image(file_path)

            return render_template('index.html', result_image=result_image_url, flower_name=flower_name)
    
    return render_template('index.html', result_image=None, flower_name=None)

def process_image(file_path):
    # Load and preprocess the image
    img = Image.open(file_path).convert('RGB')
    img = img.resize(img_size)  # Resize image to the model input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]

    # Predict the class of the image using the model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)

    # Get the predicted flower class
    predicted_flower = class_names[predicted_class[0]]

    # Save the processed image
    processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(file_path))
    img.save(processed_image_path)

    # Return the URL for the processed image and the predicted flower name
    result_image_url = url_for('static', filename=f'processed/{os.path.basename(processed_image_path)}')
    return result_image_url, predicted_flower

if __name__ == '__main__':
    app.run(debug=True)
