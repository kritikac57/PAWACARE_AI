
import os
import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def create_sample_images():
    """Create sample images for testing if they don't exist"""
    
    sample_dir = "uploads/samples"
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create a simple test image
    try:
        # Create a 224x224 RGB image with some pattern
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Add some pattern to make it look like a pet image
        cv2.rectangle(img, (50, 50), (174, 174), (100, 150, 200), -1)
        cv2.circle(img, (112, 112), 30, (200, 100, 100), -1)
        
        # Save as test image
        cv2.imwrite(os.path.join(sample_dir, "test_pet.jpg"), img)
        logger.info("Created sample test image")
        
    except Exception as e:
        logger.error(f"Error creating sample images: {e}")

def validate_image(image_path):
    """Validate if the uploaded image is valid"""
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            return False, "File does not exist"
        
        # Check file size (max 16MB)
        file_size = os.path.getsize(image_path)
        if file_size > 16 * 1024 * 1024:
            return False, "File size too large (max 16MB)"
        
        # Try to open with PIL
        with Image.open(image_path) as img:
            # Check if it's a valid image
            img.verify()
        
        # Try to read with OpenCV
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            return False, "Invalid image format"
        
        return True, "Valid image"
        
    except Exception as e:
        return False, f"Image validation error: {str(e)}"

def resize_image(image_path, target_size=(224, 224)):
    """Resize image to target size while maintaining aspect ratio"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Calculate aspect ratio
        h, w = img.shape[:2]
        aspect = w / h
        
        if aspect > 1:  # Width > Height
            new_w = target_size[0]
            new_h = int(target_size[0] / aspect)
        else:  # Height >= Width
            new_h = target_size[1]
            new_w = int(target_size[1] * aspect)
        
        # Resize image
        resized = cv2.resize(img, (new_w, new_h))
        
        # Create canvas and center the image
        canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        y_offset = (target_size[1] - new_h) // 2
        x_offset = (target_size[0] - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
        
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return None

def cleanup_old_files(directory, max_age_hours=24):
    """Clean up old uploaded files"""
    try:
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > max_age_seconds:
                    os.remove(filepath)
                    logger.info(f"Cleaned up old file: {filename}")
                    
    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")
