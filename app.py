import os
import random
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageStat
import json
from datetime import datetime

from training_module import training_bp, training_manager, DISEASE_CATEGORIES
app = Flask(__name__)
app.config['SECRET_KEY'] = 'pawcare_ai_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


# Register the training blueprint
app.register_blueprint(training_bp)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

DISEASE_CATEGORIES = {
    'healthy': {
        'name': 'Healthy',
        'description': 'No signs of disease detected. Your pet appears to be in good health.',
        'severity': 'None',
        'color': '#28a745',
        'recommendations': [
            'Continue regular veterinary checkups',
            'Maintain good hygiene and grooming',
            'Provide balanced nutrition',
            'Ensure regular exercise'
        ]
    },
    'skin_infection': {
        'name': 'Skin Infection',
        'description': 'Possible bacterial or fungal skin infection detected.',
        'severity': 'Moderate',
        'color': '#ffc107',
        'recommendations': [
            'Consult a veterinarian for proper diagnosis',
            'Keep affected area clean and dry',
            'Avoid over-bathing',
            'Consider topical treatments as prescribed'
        ]
    },
    'eye_infection': {
        'name': 'Eye Infection',
        'description': 'Signs of conjunctivitis or other eye infection detected.',
        'severity': 'Moderate',
        'color': '#fd7e14',
        'recommendations': [
            'Schedule veterinary examination immediately',
            'Clean discharge gently with warm water',
            'Prevent pet from rubbing eyes',
            'Use prescribed eye drops or ointments'
        ]
    },
    'wound': {
        'name': 'Open Wound',
        'description': 'Cut, scratch, or open wound detected.',
        'severity': 'High',
        'color': '#dc3545',
        'recommendations': [
            'Seek immediate veterinary care',
            'Clean wound with saline solution',
            'Apply pressure to control bleeding',
            'Prevent licking with cone collar'
        ]
    }
}

class SimplePawCareAI:
    def __init__(self):
        self.disease_names = list(DISEASE_CATEGORIES.keys())
    
    def analyze_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                stat = ImageStat.Stat(img)
                
                features = {
                    'brightness': sum(stat.mean) / 3,
                    'red_intensity': stat.mean[0],
                    'green_intensity': stat.mean[1],
                    'blue_intensity': stat.mean[2],
                }
                
                return features
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None
    
    def predict_disease(self, image_path):
        try:
            features = self.analyze_image(image_path)
            if features is None:
                return None
            
            # Simple rule-based prediction for demo
            probabilities = {}
            for disease in self.disease_names:
                probabilities[disease] = 0.1
            
            brightness = features['brightness']
            red_intensity = features['red_intensity']
            
            # Simple heuristics based on image properties
            if brightness > 150:
                probabilities['healthy'] += 0.5
            elif brightness < 80:
                probabilities['skin_infection'] += 0.3
            
            if red_intensity > 140:
                probabilities['skin_infection'] += 0.3
                probabilities['eye_infection'] += 0.2
                probabilities['wound'] += 0.2
                probabilities['healthy'] -= 0.2
            
            # Add some randomness for demo purposes
            for disease in probabilities:
                probabilities[disease] += random.uniform(-0.05, 0.05)
                probabilities[disease] = max(0.01, probabilities[disease])
            
            # Normalize probabilities
            total = sum(probabilities.values())
            for disease in probabilities:
                probabilities[disease] /= total
            
            predicted_disease = max(probabilities, key=probabilities.get)
            confidence = probabilities[predicted_disease]
            
            all_predictions = []
            for disease in self.disease_names:
                all_predictions.append({
                    'disease': disease,
                    'confidence': probabilities[disease],
                    'info': DISEASE_CATEGORIES[disease]
                })
            
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            result = {
                'predicted_disease': predicted_disease,
                'confidence': confidence,
                'disease_info': DISEASE_CATEGORIES[predicted_disease],
                'all_predictions': all_predictions,
                'timestamp': datetime.now().isoformat(),
                'note': 'Demo version - Always consult a veterinarian for actual diagnosis.'
            }
            
            return result
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

pawcare_ai = SimplePawCareAI()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = pawcare_ai.predict_disease(filepath)
        
        if result:
            result['filename'] = filename
            return jsonify(result)
        else:
            return jsonify({'error': 'Failed to analyze image'}), 500
            
    except Exception as e:
        print(f"Error in analyze_image: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("🐾 PawCare AI - Animal Disease Detection")
    print("Starting server...")
    print("Open your browser and go to: http://localhost:5000")

     # Initialize the training manager
    #training_manager.initialize()
    
    app.run(debug=True, host='0.0.0.0', port=5000)