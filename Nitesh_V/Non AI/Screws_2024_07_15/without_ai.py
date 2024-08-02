<<<<<<< HEAD
import cv2
import numpy as np
import os

# Path to your folder containing images
folder_path = "C:/Users/user/Downloads/Screws_2024_07_15-20240730T144942Z-001/Screws_2024_07_15"  # Replace with the correct path to your folder

# Path to the result folder
result_folder = "C:/Users/user/Downloads/Screws_2024_07_15-20240730T144942Z-001/Screws_2024_07_15/result"  # Replace with the desired path for the result folder

# Create the result folder if it doesn't exist
os.makedirs(result_folder, exist_ok=True)

# List all files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

def process_image(image_path):
    """Process a single image to detect screws and bolts."""
    # Read the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Could not load image {image_path}. Skipping.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and bounding boxes
    for contour in contours:
        # Draw the contour
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Count the number of screws
    num_screws = len(contours)
    print(f"Number of screws detected: {num_screws}")

    # Save the result image to the result folder
    result_image_path = os.path.join(result_folder, f"detected_{os.path.basename(image_path)}")
    cv2.imwrite(result_image_path, image)
    print(f"Saved result image to {result_image_path}")

# Loop through each image file
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    print(f"Processing {image_path}")
    process_image(image_path)

cv2.destroyAllWindows()
=======
import cv2
import numpy as np
import os

# Path to your folder containing images
folder_path = "C:/Users/user/Downloads/Screws_2024_07_15-20240730T144942Z-001/Screws_2024_07_15"  # Replace with the correct path to your folder

# Path to the result folder
result_folder = "C:/Users/user/Downloads/Screws_2024_07_15-20240730T144942Z-001/Screws_2024_07_15/result"  # Replace with the desired path for the result folder

# Create the result folder if it doesn't exist
os.makedirs(result_folder, exist_ok=True)

# List all files in the folder
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

def process_image(image_path):
    """Process a single image to detect screws and bolts."""
    # Read the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Could not load image {image_path}. Skipping.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and bounding boxes
    for contour in contours:
        # Draw the contour
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        # Get the bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Count the number of screws
    num_screws = len(contours)
    print(f"Number of screws detected: {num_screws}")

    # Save the result image to the result folder
    result_image_path = os.path.join(result_folder, f"detected_{os.path.basename(image_path)}")
    cv2.imwrite(result_image_path, image)
    print(f"Saved result image to {result_image_path}")

# Loop through each image file
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    print(f"Processing {image_path}")
    process_image(image_path)

cv2.destroyAllWindows()
>>>>>>> 7ba1aa2a5255f03637502b17d4a2bd29857f4ce4
