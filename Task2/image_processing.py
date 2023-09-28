import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Function to resize the images
def resize_image(image, size):
    resized_image = cv2.resize(image, size)
    return resized_image

# Function to separate colors using KMeans clustering
def separate_colors(image, num_clusters):
    # Reshape the image into a 2D array of pixels
    pixels = image.reshape(-1, 3)
    
    # Apply KMeans clustering to find the dominant colors
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(pixels)
    
    # Get the cluster labels for each pixel
    labels = kmeans.labels_
    
    # Reshape the labels back to the original image shape
    segmented_image = labels.reshape(image.shape[:2])
    
    # Get the background cluster label
    background_label = np.argmax(np.bincount(segmented_image.flatten()))
    
    # Create a mask to filter out the background pixels
    mask = np.where(segmented_image != background_label, 255, 0).astype(np.uint8)
    
    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    return masked_image, mask

# Function to crop the image based on contours
def crop_contour(image, contours):
    cropped_images = []
    for contour in contours:
        # Create a bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Crop the image using the bounding rectangle coordinates
        cropped_image = image[y:y+h, x:x+w]
        
        # Add the cropped image to the list
        cropped_images.append(cropped_image)
    
    return cropped_images

# Directory containing the images
image_directory = 'Task2\images_samples_from_camera'

# Size to resize the images
resize_size = (640, 640)

# Number of clusters for KMeans clustering
num_clusters = 3

# Read images from the directory, resize, remove background, and crop contour areas
for filename in os.listdir(image_directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_directory, filename)
        
        # Read the image
        image = cv2.imread(image_path)
        
        # Resize the image
        resized_image = resize_image(image, resize_size)
        
        # Remove background color using KMeans clustering
        masked_image, mask = separate_colors(resized_image, num_clusters)
        
        # Convert the mask to grayscale
        grayscale_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to obtain binary mask
        _, binary_mask = cv2.threshold(grayscale_image, 1, 255, cv2.THRESH_BINARY)
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crop contour areas
        cropped_images = crop_contour(resized_image, contours)
        
        # Display the cropped contour areas
        for i, cropped_image in enumerate(cropped_images):
            cv2.imshow(f'Cropped Image {i+1}', cropped_image)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()