
import cv2
import numpy as np
import os

def create_test_images():
    """Create test images for the PawCare AI application"""
    
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    # Create different types of test images
    test_images = {
        'healthy_dog.jpg': create_healthy_pet_image(),
        'skin_infection.jpg': create_skin_infection_image(),
        'eye_infection.jpg': create_eye_infection_image(),
        'test_cat.jpg': create_cat_image()
    }
    
    for filename, image in test_images.items():
        filepath = os.path.join('uploads', filename)
        cv2.imwrite(filepath, image)
        print(f"Created test image: {filepath}")

def create_healthy_pet_image():
    """Create a synthetic healthy pet image"""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Background
    img[:] = (240, 230, 220)  # Light beige background
    
    # Pet body (brown)
    cv2.ellipse(img, (200, 250), (120, 80), 0, 0, 360, (139, 69, 19), -1)
    
    # Head (lighter brown)
    cv2.circle(img, (200, 150), 60, (160, 82, 45), -1)
    
    # Eyes (healthy)
    cv2.circle(img, (180, 140), 8, (0, 0, 0), -1)  # Left eye
    cv2.circle(img, (220, 140), 8, (0, 0, 0), -1)  # Right eye
    cv2.circle(img, (182, 138), 3, (255, 255, 255), -1)  # Eye shine
    cv2.circle(img, (222, 138), 3, (255, 255, 255), -1)  # Eye shine
    
    # Nose
    cv2.ellipse(img, (200, 160), (6, 4), 0, 0, 360, (0, 0, 0), -1)
    
    # Ears
    cv2.ellipse(img, (160, 120), (20, 35), -30, 0, 360, (139, 69, 19), -1)
    cv2.ellipse(img, (240, 120), (20, 35), 30, 0, 360, (139, 69, 19), -1)
    
    return img

def create_skin_infection_image():
    """Create a synthetic skin infection image"""
    img = create_healthy_pet_image()
    
    # Add red, inflamed patches
    cv2.circle(img, (150, 200), 25, (50, 50, 200), -1)  # Red patch
    cv2.circle(img, (250, 220), 20, (40, 40, 180), -1)  # Another red patch
    
    # Add some texture to simulate infection
    for i in range(10):
        x = np.random.randint(130, 170)
        y = np.random.randint(180, 220)
        cv2.circle(img, (x, y), 3, (30, 30, 150), -1)
    
    return img

def create_eye_infection_image():
    """Create a synthetic eye infection image"""
    img = create_healthy_pet_image()
    
    # Make one eye red and inflamed
    cv2.circle(img, (180, 140), 15, (50, 50, 200), -1)  # Red around eye
    cv2.circle(img, (180, 140), 8, (100, 100, 255), -1)  # Inflamed eye
    
    # Add discharge
    cv2.ellipse(img, (175, 150), (8, 4), 0, 0, 360, (200, 200, 100), -1)
    
    return img

def create_cat_image():
    """Create a synthetic cat image"""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Background
    img[:] = (245, 245, 245)  # Light gray background
    
    # Cat body (gray)
    cv2.ellipse(img, (200, 280), (100, 70), 0, 0, 360, (128, 128, 128), -1)
    
    # Head (gray)
    cv2.circle(img, (200, 180), 50, (128, 128, 128), -1)
    
    # Ears (triangular)
    triangle1 = np.array([[170, 140], [160, 110], [180, 120]], np.int32)
    triangle2 = np.array([[230, 140], [240, 110], [220, 120]], np.int32)
    cv2.fillPoly(img, [triangle1], (128, 128, 128))
    cv2.fillPoly(img, [triangle2], (128, 128, 128))
    
    # Eyes (cat-like)
    cv2.ellipse(img, (185, 170), (8, 12), 0, 0, 360, (0, 255, 0), -1)  # Green eyes
    cv2.ellipse(img, (215, 170), (8, 12), 0, 0, 360, (0, 255, 0), -1)
    cv2.ellipse(img, (185, 170), (2, 8), 0, 0, 360, (0, 0, 0), -1)  # Pupils
    cv2.ellipse(img, (215, 170), (2, 8), 0, 0, 360, (0, 0, 0), -1)
    
    # Nose (pink triangle)
    triangle_nose = np.array([[200, 185], [195, 195], [205, 195]], np.int32)
    cv2.fillPoly(img, [triangle_nose], (255, 192, 203))
    
    # Whiskers
    cv2.line(img, (150, 185), (120, 180), (0, 0, 0), 2)
    cv2.line(img, (150, 195), (120, 195), (0, 0, 0), 2)
    cv2.line(img, (250, 185), (280, 180), (0, 0, 0), 2)
    cv2.line(img, (250, 195), (280, 195), (0, 0, 0), 2)
    
    return img

if __name__ == "__main__":
    create_test_images()
    print("Test images created successfully!")
    print("You can now test the application with these images:")
    print("- uploads/healthy_dog.jpg")
    print("- uploads/skin_infection.jpg") 
    print("- uploads/eye_infection.jpg")
    print("- uploads/test_cat.jpg")
