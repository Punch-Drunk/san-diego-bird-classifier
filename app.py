import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image
import tensorflow as tf
import io
import pandas as pd
import base64

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('models/bird_classifier.keras')

# Load class names from training directory
df = pd.read_csv('SD Birds.csv')
class_names = sorted(df['Bird Names'].tolist()) 

# Create a mapping from scientific to common names
def reformat_common_name(name):
    if ', ' in name:
        parts = name.split(', ', 1)
        return f"{parts[1].strip()} {parts[0].strip()}"
    return name

# Load the mapping from CSV
import pandas as pd
df = pd.read_csv('SD Birds.csv')
COMMON_NAMES = {}
for _, row in df.iterrows():
    COMMON_NAMES[row['Bird Names']] = reformat_common_name(row['Common Name'])

def preprocess_image(image):
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to match model's expected input
    image = image.resize((256, 256))
    
    # Convert to numpy array and normalize
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    
    return img_array

def get_prediction(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Get model predictions
    predictions = model.predict(processed_image)
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    results = []
    
    for idx in top_3_idx:
        scientific_name = class_names[idx]
        common_name = COMMON_NAMES.get(scientific_name, scientific_name)
        confidence = float(predictions[0][idx])
        results.append({
            'common_name': common_name,
            'scientific_name': scientific_name,
            'confidence': f"{confidence:.2%}"
        })
    
    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        image_data = request.json['image']
        # Remove the data URL prefix
        image_data = image_data.split(',')[1]
        
        # Decode base64 image
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # Get predictions
        results = get_prediction(image)
        
        return jsonify({'success': True, 'predictions': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 