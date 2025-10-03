
from flask import Blueprint, request, jsonify, send_file
import os
import json
from datetime import datetime
from training_module import TrainingManager, log_training_event, get_recent_training_logs
import threading
import zipfile
import tempfile

api_bp = Blueprint('api', __name__, url_prefix='/api')
training_manager = TrainingManager()

@api_bp.route('/training/start', methods=['POST'])
def start_training():
    """Start a new training session"""
    try:
        data = request.get_json()
        
        # Validate parameters
        epochs = data.get('epochs', 50)
        batch_size = data.get('batch_size', 32)
        learning_rate = data.get('learning_rate', 0.001)
        
        if epochs < 1 or epochs > 200:
            return jsonify({'success': False, 'error': 'Epochs must be between 1 and 200'})
        
        if batch_size not in [16, 32, 64]:
            return jsonify({'success': False, 'error': 'Batch size must be 16, 32, or 64'})
        
        if learning_rate not in [0.0001, 0.001, 0.01]:
            return jsonify({'success': False, 'error': 'Learning rate must be 0.0001, 0.001, or 0.01'})
        
        # Check if training is already in progress
        if training_manager.is_training():
            return jsonify({'success': False, 'error': 'Training is already in progress'})
        
        # Validate dataset
        validation_result = training_manager.validate_dataset()
        if not validation_result['is_valid']:
            return jsonify({
                'success': False, 
                'error': 'Dataset validation failed',
                'issues': validation_result['issues']
            })
        
        # Start training in background thread
        training_config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        training_thread = threading.Thread(
            target=training_manager.start_training,
            args=(training_config,)
        )
        training_thread.daemon = True
        training_thread.start()
        
        log_training_event(f"Training started with config: {training_config}")
        
        return jsonify({'success': True, 'message': 'Training started successfully'})
        
    except Exception as e:
        log_training_event(f"Error starting training: {str(e)}", level='ERROR')
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/training/stop', methods=['POST'])
def stop_training():
    """Stop the current training session"""
    try:
        if not training_manager.is_training():
            return jsonify({'success': False, 'error': 'No training in progress'})
        
        training_manager.stop_training()
        log_training_event("Training stopped by user")
        
        return jsonify({'success': True, 'message': 'Training stopped successfully'})
        
    except Exception as e:
        log_training_event(f"Error stopping training: {str(e)}", level='ERROR')
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/training/status', methods=['GET'])
def get_training_status():
    """Get current training status"""
    try:
        status = training_manager.get_training_status()
        return jsonify({'success': True, 'status': status})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/training/progress', methods=['GET'])
def get_training_progress():
    """Get real-time training progress"""
    try:
        progress = training_manager.get_training_progress()
        return jsonify({'success': True, 'progress': progress})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/dataset/validate', methods=['POST'])
def validate_dataset():
    """Validate the training dataset"""
    try:
        validation_result = training_manager.validate_dataset()
        return jsonify({'success': True, 'validation': validation_result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/dataset/statistics', methods=['GET'])
def get_dataset_statistics():
    """Get dataset statistics"""
    try:
        stats = training_manager.get_dataset_statistics()
        return jsonify({'success': True, 'statistics': stats})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/dataset/export', methods=['GET'])
def export_dataset():
    """Export the training dataset as a ZIP file"""
    try:
        # Create temporary ZIP file
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, f'training_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip')
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add training images
            training_data_dir = os.path.join('static', 'training_data')
            if os.path.exists(training_data_dir):
                for root, dirs, files in os.walk(training_data_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, training_data_dir)
                            zipf.write(file_path, arcname)
            
            # Add metadata
            metadata = {
                'export_date': datetime.now().isoformat(),
                'statistics': training_manager.get_dataset_statistics(),
                'categories': training_manager.get_categories()
            }
            zipf.writestr('metadata.json', json.dumps(metadata, indent=2))
        
        log_training_event("Dataset exported")
        
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=os.path.basename(zip_path),
            mimetype='application/zip'
        )
        
    except Exception as e:
        log_training_event(f"Error exporting dataset: {str(e)}", level='ERROR')
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/models/list', methods=['GET'])
def list_models():
    """Get list of all trained models"""
    try:
        models = training_manager.get_models()
        return jsonify({'success': True, 'models': models})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/models/activate/<model_id>', methods=['POST'])
def activate_model(model_id):
    """Activate a specific model"""
    try:
        success = training_manager.activate_model(model_id)
        if success:
            log_training_event(f"Model {model_id} activated")
            return jsonify({'success': True, 'message': 'Model activated successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to activate model'})
            
    except Exception as e:
        log_training_event(f"Error activating model {model_id}: {str(e)}", level='ERROR')
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/models/delete/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a specific model"""
    try:
        success = training_manager.delete_model(model_id)
        if success:
            log_training_event(f"Model {model_id} deleted")
            return jsonify({'success': True, 'message': 'Model deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to delete model'})
            
    except Exception as e:
        log_training_event(f"Error deleting model {model_id}: {str(e)}", level='ERROR')
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/logs/recent', methods=['GET'])
def get_recent_logs():
    """Get recent training logs"""
    try:
        limit = request.args.get('limit', 10, type=int)
        logs = get_recent_training_logs(limit)
        return jsonify({'success': True, 'logs': logs})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/logs/download', methods=['GET'])
def download_logs():
    """Download training logs as a text file"""
    try:
        log_file = os.path.join('logs', 'training.log')
        if os.path.exists(log_file):
            return send_file(
                log_file,
                as_attachment=True,
                download_name=f'training_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
                mimetype='text/plain'
            )
        else:
            return jsonify({'success': False, 'error': 'Log file not found'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Health check endpoint
@api_bp.route('/health', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'training_active': training_manager.is_training()
    })

from flask import Blueprint, request, jsonify, send_file
import os
import json
from datetime import datetime
from training_module import TrainingManager, log_training_event, get_recent_training_logs
import threading
import zipfile
import tempfile

api_bp = Blueprint('api', __name__, url_prefix='/api')
training_manager = TrainingManager()

@api_bp.route('/training/start', methods=['POST'])
def start_training():
    """Start a new training session"""
    try:
        data = request.get_json()
        
        # Validate parameters
        epochs = data.get('epochs', 50)
        batch_size = data.get('batch_size', 32)
        learning_rate = data.get('learning_rate', 0.001)
        
        if epochs < 1 or epochs > 200:
            return jsonify({'success': False, 'error': 'Epochs must be between 1 and 200'})
        
        if batch_size not in [16, 32, 64]:
            return jsonify({'success': False, 'error': 'Batch size must be 16, 32, or 64'})
        
        if learning_rate not in [0.0001, 0.001, 0.01]:
            return jsonify({'success': False, 'error': 'Learning rate must be 0.0001, 0.001, or 0.01'})
        
        # Check if training is already in progress
        if training_manager.is_training():
            return jsonify({'success': False, 'error': 'Training is already in progress'})
        
        # Validate dataset
        validation_result = training_manager.validate_dataset()
        if not validation_result['is_valid']:
            return jsonify({
                'success': False, 
                'error': 'Dataset validation failed',
                'issues': validation_result['issues']
            })
        
        # Start training in background thread
        training_config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'timestamp': datetime.now().isoformat()
        }
        
        training_thread = threading.Thread(
            target=training_manager.start_training,
            args=(training_config,)
        )
        training_thread.daemon = True
        training_thread.start()
        
        log_training_event(f"Training started with config: {training_config}")
        
        return jsonify({'success': True, 'message': 'Training started successfully'})
        
    except Exception as e:
        log_training_event(f"Error starting training: {str(e)}", level='ERROR')
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/training/stop', methods=['POST'])
def stop_training():
    """Stop the current training session"""
    try:
        if not training_manager.is_training():
            return jsonify({'success': False, 'error': 'No training in progress'})
        
        training_manager.stop_training()
        log_training_event("Training stopped by user")
        
        return jsonify({'success': True, 'message': 'Training stopped successfully'})
        
    except Exception as e:
        log_training_event(f"Error stopping training: {str(e)}", level='ERROR')
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/training/status', methods=['GET'])
def get_training_status():
    """Get current training status"""
    try:
        status = training_manager.get_training_status()
        return jsonify({'success': True, 'status': status})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/training/progress', methods=['GET'])
def get_training_progress():
    """Get real-time training progress"""
    try:
        progress = training_manager.get_training_progress()
        return jsonify({'success': True, 'progress': progress})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/dataset/validate', methods=['POST'])
def validate_dataset():
    """Validate the training dataset"""
    try:
        validation_result = training_manager.validate_dataset()
        return jsonify({'success': True, 'validation': validation_result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/dataset/statistics', methods=['GET'])
def get_dataset_statistics():
    """Get dataset statistics"""
    try:
        stats = training_manager.get_dataset_statistics()
        return jsonify({'success': True, 'statistics': stats})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/dataset/export', methods=['GET'])
def export_dataset():
    """Export the training dataset as a ZIP file"""
    try:
        # Create temporary ZIP file
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, f'training_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip')
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add training images
            training_data_dir = os.path.join('static', 'training_data')
            if os.path.exists(training_data_dir):
                for root, dirs, files in os.walk(training_data_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, training_data_dir)
                            zipf.write(file_path, arcname)
            
            # Add metadata
            metadata = {
                'export_date': datetime.now().isoformat(),
                'statistics': training_manager.get_dataset_statistics(),
                'categories': training_manager.get_categories()
            }
            zipf.writestr('metadata.json', json.dumps(metadata, indent=2))
        
        log_training_event("Dataset exported")
        
        return send_file(
            zip_path,
            as_attachment=True,
            download_name=os.path.basename(zip_path),
            mimetype='application/zip'
        )
        
    except Exception as e:
        log_training_event(f"Error exporting dataset: {str(e)}", level='ERROR')
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/models/list', methods=['GET'])
def list_models():
    """Get list of all trained models"""
    try:
        models = training_manager.get_models()
        return jsonify({'success': True, 'models': models})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/models/activate/<model_id>', methods=['POST'])
def activate_model(model_id):
    """Activate a specific model"""
    try:
        success = training_manager.activate_model(model_id)
        if success:
            log_training_event(f"Model {model_id} activated")
            return jsonify({'success': True, 'message': 'Model activated successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to activate model'})
            
    except Exception as e:
        log_training_event(f"Error activating model {model_id}: {str(e)}", level='ERROR')
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/models/delete/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a specific model"""
    try:
        success = training_manager.delete_model(model_id)
        if success:
            log_training_event(f"Model {model_id} deleted")
            return jsonify({'success': True, 'message': 'Model deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to delete model'})
            
    except Exception as e:
        log_training_event(f"Error deleting model {model_id}: {str(e)}", level='ERROR')
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/logs/recent', methods=['GET'])
def get_recent_logs():
    """Get recent training logs"""
    try:
        limit = request.args.get('limit', 10, type=int)
        logs = get_recent_training_logs(limit)
        return jsonify({'success': True, 'logs': logs})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@api_bp.route('/logs/download', methods=['GET'])
def download_logs():
    """Download training logs as a text file"""
    try:
        log_file = os.path.join('logs', 'training.log')
        if os.path.exists(log_file):
            return send_file(
                log_file,
                as_attachment=True,
                download_name=f'training_logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
                mimetype='text/plain'
            )
        else:
            return jsonify({'success': False, 'error': 'Log file not found'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Health check endpoint
@api_bp.route('/health', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'training_active': training_manager.is_training()
    })
