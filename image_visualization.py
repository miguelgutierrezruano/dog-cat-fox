
from include import *

def visualize_images(path, num_images=5):
    # Get a list of image filenames in the specified path
    image_filenames = os.listdir(path)
    
    # Limit the number of images to visualize if there are more than num_images
    num_images = min(num_images, len(image_filenames))
    
    # Create a figure and axis object to display images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3),facecolor='darkblue')
    
    # Iterate over the selected images and display them
    for i, image_filename in enumerate(image_filenames[:num_images]):
        # Load the image using Matplotlib
        image_path = os.path.join(path, image_filename)
        image = mpimg.imread(image_path)
        
        # Display the image
        axes[i].imshow(image)
        axes[i].axis('off')  # Turn off axis
        axes[i].set_title(image_filename)  # Set image filename as title
    
    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()

# Specify the path containing the images to visualize
path_to_visualize = "pic-dataset/train/cats"

# Visualize some images from the specified path
visualize_images(path_to_visualize, num_images=5)

# Specify the path containing the images to visualize
path_to_visualize = "pic-dataset/train/dogs" 

# Visualize some images from the specified path
visualize_images(path_to_visualize, num_images=5)

# Specify the path containing the images to visualize
path_to_visualize = "pic-dataset/train/foxes" 

# Visualize some images from the specified path
visualize_images(path_to_visualize, num_images=5)