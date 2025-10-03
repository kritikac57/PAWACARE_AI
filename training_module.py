
import os
import json
import time
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
import numpy as np
from PIL import Image
import shutil

# Create blueprint for training module
training_bp = Blueprint('training', __name__, url_prefix='/training')

# Training configuration
TRAINING_DATA_DIR = 'training_data'
MODELS_DIR = 'models'
TRAINING_LOGS_DIR = 'training_logs'

# Create necessary directories
for directory in [TRAINING_DATA_DIR, MODELS_DIR, TRAINING_LOGS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Disease categories for training
DISEASE_CATEGORIES = [
    'healthy',
    'skin_infection',
    'eye_infection',
    'ear_infection',
    'dental_issues',
    'respiratory_issues',
    'digestive_issues',
    'parasites',
    'wounds_injuries',
    'allergic_reactions'
]

class TrainingManager:
    def __init__(self):
        self.training_status = {
            'is_training': False,
            'current_epoch': 0,
            'total_epochs': 0,
            'current_loss': 0.0,
            'current_accuracy': 0.0,
            'start_time': None,
            'estimated_completion': None
        }
        
    def get_dataset_stats(self):
        """Get statistics about the training dataset"""
        stats = {}
        total_images = 0
        
        for category in DISEASE_CATEGORIES:
            category_path = os.path.join(TRAINING_DATA_DIR, category)
            if os.path.exists(category_path):
                image_count = len([f for f in os.listdir(category_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
                stats[category] = image_count
                total_images += image_count
            else:
                stats[category] = 0
                
        stats['total'] = total_images
        return stats
    
    def validate_training_data(self):
        """Validate training data structure and quality"""
        issues = []
        recommendations = []
        
        stats = self.get_dataset_stats()
        
        # Check minimum images per category
        min_images_per_category = 50
        for category, count in stats.items():
            if category != 'total' and count < min_images_per_category:
                issues.append(f"Category '{category}' has only {count} images (minimum recommended: {min_images_per_category})")
        
        # Check class imbalance
        if stats['total'] > 0:
            category_counts = [count for category, count in stats.items() if category != 'total']
            if category_counts:
                max_count = max(category_counts)
                min_count = min(category_counts)
                if max_count > 0 and (max_count / min_count) > 5:
                    issues.append("Significant class imbalance detected (some categories have 5x more images than others)")
                    recommendations.append("Consider balancing your dataset by adding more images to underrepresented categories")
        
        # Check total dataset size
        if stats['total'] < 500:
            recommendations.append("Consider adding more training data for better model performance (current: {}, recommended: 500+)".format(stats['total']))
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'is_valid': len(issues) == 0
        }

# Initialize training manager
training_manager = TrainingManager()

@training_bp.route('/')
def dashboard():
    """Training dashboard"""
    stats = training_manager.get_dataset_stats()
    validation = training_manager.validate_training_data()
    
    # Get recent training logs
    recent_logs = get_recent_training_logs(5)
    
    return render_template('training/dashboard.html', 
                         stats=stats, 
                         validation=validation,
                         recent_logs=recent_logs,
                         categories=DISEASE_CATEGORIES,
                         training_status=training_manager.training_status)

@training_bp.route('/upload_data')
def upload_data():
    """Upload training data page"""
    return render_template('training/upload_data.html', categories=DISEASE_CATEGORIES)

@training_bp.route('/api/upload_training_image', methods=['POST'])
def upload_training_image():
    """API endpoint to upload training images"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        category = request.form.get('category')
        
        if not category or category not in DISEASE_CATEGORIES:
            return jsonify({'error': 'Invalid category'})
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            return jsonify({'error': 'Invalid file type'})
        
        # Create category directory if it doesn't exist
        category_dir = os.path.join(TRAINING_DATA_DIR, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        
        # Generate unique filename
        timestamp = int(time.time())
        filename = f"{category}_{timestamp}_{file.filename}"
        filepath = os.path.join(category_dir, filename)
        
        # Save and validate image
        file.save(filepath)
        
        try:
            with Image.open(filepath) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    img.save(filepath)
                
                # Get image info
                width, height = img.size
                
        except Exception as e:
            os.remove(filepath)
            return jsonify({'error': 'Invalid image file'})
        
        return jsonify({
            'success': True,
            'message': f'Image uploaded successfully to {category} category',
            'filename': filename,
            'dimensions': f"{width}x{height}"
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'})

@training_bp.route('/api/start_training', methods=['POST'])
def start_training():
    """Start model training"""
    try:
        if training_manager.training_status['is_training']:
            return jsonify({'error': 'Training is already in progress'})
        
        # Get training parameters
        data = request.get_json()
        epochs = data.get('epochs', 50)
        batch_size = data.get('batch_size', 32)
        learning_rate = data.get('learning_rate', 0.001)
        
        # Validate training data
        validation = training_manager.validate_training_data()
        if not validation['is_valid']:
            return jsonify({
                'error': 'Training data validation failed',
                'issues': validation['issues']
            })
        
        # Start training (mock implementation)
        training_id = start_mock_training(epochs, batch_size, learning_rate)
        
        return jsonify({
            'success': True,
            'message': 'Training started successfully',
            'training_id': training_id
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to start training: {str(e)}'})

@training_bp.route('/api/training_status')
def get_training_status():
    """Get current training status"""
    return jsonify(training_manager.training_status)

@training_bp.route('/api/stop_training', methods=['POST'])
def stop_training():
    """Stop current training"""
    try:
        if not training_manager.training_status['is_training']:
            return jsonify({'error': 'No training in progress'})
        
        # Stop training (mock implementation)
        training_manager.training_status['is_training'] = False
        
        # Log training stop
        log_training_event('Training stopped by user')
        
        return jsonify({
            'success': True,
            'message': 'Training stopped successfully'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to stop training: {str(e)}'})

@training_bp.route('/models')
def models():
    """Manage trained models"""
    model_list = get_available_models()
    return render_template('training/models.html', models=model_list)

@training_bp.route('/api/delete_model/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a trained model"""
    try:
        model_path = os.path.join(MODELS_DIR, f"{model_id}.pkl")
        if os.path.exists(model_path):
            os.remove(model_path)
            log_training_event(f'Model {model_id} deleted')
            return jsonify({'success': True, 'message': 'Model deleted successfully'})
        else:
            return jsonify({'error': 'Model not found'})
    except Exception as e:
        return jsonify({'error': f'Failed to delete model: {str(e)}'})

@training_bp.route('/logs')
def logs():
    """View training logs"""
    logs = get_recent_training_logs(50)
    return render_template('training/logs.html', logs=logs)

# Helper functions
def start_mock_training(epochs, batch_size, learning_rate):
    """Mock training function (replace with actual training logic)"""
    training_id = f"training_{int(time.time())}"
    
    # Update training status
    training_manager.training_status.update({
        'is_training': True,
        'current_epoch': 0,
        'total_epochs': epochs,
        'current_loss': 1.0,
        'current_accuracy': 0.0,
        'start_time': datetime.now().isoformat(),
        'estimated_completion': None
    })
    
    # Log training start
    log_training_event(f'Training started: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}')
    
    # In a real implementation, you would start actual training here
    # For now, we'll simulate training progress
    
    return training_id

def get_available_models():
    """Get list of available trained models"""
    models = []
    if os.path.exists(MODELS_DIR):
        for filename in os.listdir(MODELS_DIR):
            if filename.endswith('.pkl'):
                model_path = os.path.join(MODELS_DIR, filename)
                stat = os.stat(model_path)
                models.append({
                    'id': filename[:-4],  # Remove .pkl extension
                    'name': filename,
                    'size': f"{stat.st_size / 1024 / 1024:.2f} MB",
                    'created': datetime.fromtimestamp(stat.st_ctime).strftime('%Y-%m-%d %H:%M:%S'),
                    'path': model_path
                })
    return models

def log_training_event(message):
    """Log training events"""
    log_file = os.path.join(TRAINING_LOGS_DIR, 'training.log')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")

def get_recent_training_logs(limit=10):
    """Get recent training log entries"""
    log_file = os.path.join(TRAINING_LOGS_DIR, 'training.log')
    logs = []
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-limit:]:
                logs.append(line.strip())
    return logs
def get_recent_training_logs(limit=10):
    """Get recent training log entries"""
    log_file = os.path.join(TRAINING_LOGS_DIR, 'training.log')
    logs = []
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            # Get last 'limit' lines
            recent_lines = lines[-limit:] if len(lines) > limit else lines
            
            for line in recent_lines:
                line = line.strip()
                if line:
                    # Parse log format: [timestamp] message
                    if line.startswith('[') and ']' in line:
                        timestamp_end = line.find(']')
                        timestamp = line[1:timestamp_end]
                        message = line[timestamp_end + 2:]
                        logs.append({
                            'timestamp': timestamp,
                            'message': message
                        })
    
    return list(reversed(logs))  # Most recent first