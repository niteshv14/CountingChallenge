import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, filters
import os

# Path to the input folder
folder_path = 'C:/Users/user/Downloads/ScrewAndBolt_20240713-20240730T144943Z-001/ScrewAndBolt_20240713'
# Path to the output folder
result_folder = 'C:/Users/user/Downloads/ScrewAndBolt_20240713-20240730T144943Z-001/ScrewAndBolt_20240713/result'

# Create result folder if it doesn't exist
os.makedirs(result_folder, exist_ok=True)

# Parameters
min_area = 100  # Minimum area to keep for filtering
block_size = 35
offset = 10

# List all files in the folder
files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]

for file_name in files:
    image_path = os.path.join(folder_path, file_name)
    
    # Task 1: Pre-processing -----------------------
    # Step-1: Load input image
    I = cv2.imread(image_path)
    I_rgb = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(I_rgb)
    plt.title(f'Step-1: Input Image - {file_name}')
    plt.axis('off')
    #plt.savefig(os.path.join(result_folder, f'{file_name}_step1.png'), bbox_inches='tight')
    plt.close()  

    # Step-2: Convert image to grayscale
    I_gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

    plt.figure()
    plt.imshow(I_gray, cmap='gray')
    plt.title('Step-2: Grayscale Image')
    plt.axis('off')
    #plt.savefig(os.path.join(result_folder, f'{file_name}_step2.png'), bbox_inches='tight')
    plt.close()  

    # Step-3: Rescale image by bilinear interpolation
    I_scaled = cv2.resize(I_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    plt.figure()
    plt.imshow(I_scaled, cmap='gray')
    plt.title('Step-3: Rescaled by Linear Interpolation Image')
    plt.axis('off')
    #plt.savefig(os.path.join(result_folder, f'{file_name}_step3.png'), bbox_inches='tight')
    plt.close()  

    # Step-4: Produce histogram before enhancing
    plt.figure()
    plt.hist(I_scaled.ravel(), bins=256, range=(0, 256))
    plt.title('Step-4: Histogram Before Enhancing')
    #plt.savefig(os.path.join(result_folder, f'{file_name}_step4_hist.png'), bbox_inches='tight')
    plt.close()  

    # Step-5: Enhance image before binarization
    I_enhanced = cv2.equalizeHist(I_scaled)

    plt.figure()
    plt.imshow(I_enhanced, cmap='gray')
    plt.title('Step-5: Enhanced Image Before Binarisation')
    plt.axis('off')
    plt.savefig(os.path.join(result_folder, f'{file_name}_step5.png'), bbox_inches='tight')
    plt.close()  

    # Step-6: Histogram after enhancement
    plt.figure()
    plt.hist(I_enhanced.ravel(), bins=256, range=(0, 256))
    plt.title('Step-6: Histogram After Enhancement')
    #plt.savefig(os.path.join(result_folder, f'{file_name}_step6_hist.png'), bbox_inches='tight')
    plt.close()  

    # Step-7: Image Binarisation
    I_binarised = filters.threshold_local(I_enhanced, block_size=block_size, offset=offset)
    I_binarised = I_enhanced > I_binarised

    plt.figure()
    plt.imshow(I_binarised, cmap='gray')
    plt.title('Step-7: Binarised Image')
    plt.axis('off')
    plt.savefig(os.path.join(result_folder, f'{file_name}_step7.png'), bbox_inches='tight')
    plt.close()  
    
    # Remove small objects
    # Label connected components
    I_labeled = measure.label(I_binarised, connectivity=2)
    
    # Calculate the properties of labeled regions
    properties = measure.regionprops(I_labeled)
    
    # Filter out small objects based on area
    filtered_labels = np.zeros_like(I_labeled)
    for prop in properties:
        if prop.area >= min_area:
            filtered_labels[I_labeled == prop.label] = prop.label
    
    # Count objects in the filtered image
    num_objects = len(np.unique(filtered_labels)) - 1  # Subtract 1 to ignore the background label
    
    # Ensure correct object count by ignoring background
    num_objects = len(np.unique(filtered_labels[filtered_labels > 0]))  # Only count non-background objects
    
    print(f'Number of objects in {file_name} after filtering: {num_objects}')
    
    # Display filtered labeled image
    plt.figure()
    plt.imshow(filtered_labels, cmap='nipy_spectral')
    plt.title(f'Filtered Labeled Image - {file_name}')
    plt.axis('off')
    plt.savefig(os.path.join(result_folder, f'{file_name}_filtered_labels.png'), bbox_inches='tight')

plt.show()
