
# Imported code from https://www.kaggle.com/code/abdmental01/cat-vs-dog-transfer-learning-0-99/notebook

#Checking GPU Support
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

#Import Os and Basis Libraries
import cv2
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
#Matplot Images
import matplotlib.image as mpimg
# Tensflor and Keras Layer and Model and Optimize and Loss
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import *
from tensorflow.keras.losses import BinaryCrossentropy
# import tensorflow_hub as hub
from tensorflow.keras.optimizers import Adam
#PreTrained Model VGG16
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import Xception
#Image Generator DataAugmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
#Early Stopping
from tensorflow.keras.callbacks import EarlyStopping
# Warnings Remove 
import warnings 
warnings.filterwarnings("ignore")

# Directory containing the "Train" folder
directory = "dogs_vs_cats"

# List of categories (subfolder names)
categories = ["cats", "dogs"]

# Initialize lists to store filenames and categories
filenames = []
category_labels = []

# Iterate through the categories
for category in categories:
    # Path to the current category folder
    category_folder = os.path.join(directory, "train", category)
    # List all filenames in the category folder
    category_filenames = os.listdir(category_folder)
    # Append filenames and corresponding category labels
    filenames.extend(category_filenames)
    category_labels.extend([category] * len(category_filenames))

# Create DataFrame
df = pd.DataFrame({
    'filename': filenames,
    'category': category_labels
})

# Display the first few rows of the DataFrame
print(df.head())

# Count the occurrences of each category in the 'category' column
count = df['category'].value_counts()

# Create a pie chart using Seaborn
plt.figure(figsize=(6, 6) , facecolor='darkblue')
palette = sns.color_palette("viridis")
sns.set_palette(palette)
plt.pie(count, labels=count.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Categories') 
plt.axis('equal') 

plt.show()  # Show the plot

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
path_to_visualize = "dogs_vs_cats/train/cats"

# Visualize some images from the specified path
visualize_images(path_to_visualize, num_images=5)

# Specify the path containing the images to visualize
path_to_visualize = "dogs_vs_cats/train/dogs" 

# Visualize some images from the specified path
visualize_images(path_to_visualize, num_images=5)

#Data_Dir
data_dir = 'dogs_vs_cats/train'

# Defining data generator with Data Augmentation
data_gen_augmented = ImageDataGenerator(rescale = 1/255., 
                                        validation_split = 0.2,
                                        zoom_range = 0.2,
                                        horizontal_flip= True,
                                        rotation_range = 20,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2)
print('Augmented training Images:')
train_ds = data_gen_augmented.flow_from_directory(data_dir, 
                                                              target_size = (224, 224), 
                                                              batch_size = 32,
                                                              subset = 'training',
                                                              class_mode = 'binary')

#Testing Augmented Data
# Defining Validation_generator withour Data Augmentation
data_gen = ImageDataGenerator(rescale = 1/255., validation_split = 0.2)

print('Unchanged Validation Images:')
validation_ds = data_gen.flow_from_directory(data_dir, 
                                        target_size = (224, 224), 
                                        batch_size = 32,
                                        subset = 'validation',
                                        class_mode = 'binary')

# Load the pre-trained Xception model without the top (classification) layer
xception_base = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
xception_base.trainable = False

# Build the model
model = Sequential()

# Add the pre-trained Xception base
model.add(xception_base)

# Add global average pooling layer to reduce spatial dimensions
model.add(AveragePooling2D(pool_size=(2, 2)))

# Flatten the feature maps
model.add(Flatten())

# Add a dense layer with 220 units and ReLU activation function
model.add(Dense(220, activation='relu'))

# Add the output layer with 1 unit and sigmoid activation function for binary classification
model.add(Dense(1, activation='sigmoid'))

model.summary()

# Compile
model.compile(loss = BinaryCrossentropy(),
                optimizer = keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.9),
                metrics = ['accuracy'])

#Early_Stopping
early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=5, 
    restore_best_weights=True,
)

#Fitting Model
history = model.fit(train_ds,
                        epochs= 10,
                        steps_per_epoch = len(train_ds),
                        validation_data = validation_ds,
                        validation_steps = len(validation_ds),
                        callbacks = early_stopping)

# Evaluate the model on the validation dataset
validation_loss, validation_accuracy = model.evaluate(validation_ds)

# Print the validation loss and accuracy
print("Validation Loss:", validation_loss)
print("Validation Accuracy:", validation_accuracy)

# Accuracy and Val_Accuracy
plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

# Loss and Val_Loss
plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()

# Get the class indices assigned by the generators
class_indices_train = train_ds.class_indices

# Print the class indices
print("Class indices for training generator:", class_indices_train)

#Testing Augmented Data
test_dir_path = "archive/test"
# Defining Validation_generator withour Data Augmentation
data_test_gen = ImageDataGenerator(rescale = 1/255.)

print('Test Validation Images:')
test_ds = data_gen.flow_from_directory(test_dir_path, 
                                        target_size = (224, 224), 
                                        batch_size = 32,
                                        subset = 'validation',
                                        class_mode = 'binary')

# Evaluate the model on the validation dataset
test_loss, test_accuracy = model.evaluate(test_ds)

# Print the validation loss and accuracy
print("Test Loss:", validation_loss)
print("Test Accuracy:", validation_accuracy)

print(f'Well Our Model is Performing Well On Unseen Data With Accuracy of : {test_accuracy} \n Which is Actually a Good Performence.')

# Initialize the accuracies
acc_train = history.history['accuracy']
val_acc = validation_accuracy
test_acc = test_accuracy

# Create a DataFrame for the accuracies
data = {
    'Accuracy Type': ['Train Accuracy', 'Validation Accuracy', 'Test Accuracy'],
    'Accuracy': [acc_train[-1], val_acc, test_acc]
}
df = pd.DataFrame(data)

# Set the background color
background_color = 'darkblue'

# Set the color palette
palette = 'viridis'

# Set Seaborn style and color palette
sns.set_style("darkgrid")
sns.set_palette(palette)

# Create the bar plot
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(x='Accuracy Type', y='Accuracy', data=df, ci=None)

# Set background color
bar_plot.set_facecolor(background_color)

# Add title and labels
plt.title("Model Accuracies")
plt.xlabel("Accuracy Type")
plt.ylabel("Accuracy")

# Show the plot
plt.show()

# List of paths to your single images
image_paths = ['archive/train/cats/cat.0.jpg', 'archive/train/cats/cat.1.jpg', 'archive/train/dogs/dog.1.jpg']
# Intialize true labels
true_labels = ['Cat', 'Cat','Dog']

# Load and preprocess each image, make predictions, and display them using a loop
for img_path, true_label in zip(image_paths, true_labels):
    # Load the image using OpenCV
    img = cv2.imread(img_path)
    # Resize the image to (224, 224)
    img = cv2.resize(img, (224, 224)) 

    # Normalize pixel values
    img_array = img.astype(np.float32) / 255.0  

    # Expand the dimensions to match the input shape expected by the model
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions
    predictions = model.predict(img_array)
    actual_prediction = (predictions > 0.5).astype(int)

    # Display the image with true and predicted labels
    # Convert BGR to RGB for displaying with matplotlib
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  
    plt.axis('off')
    if actual_prediction[0][0] == 0:
        predicted_label = 'Cat'
    else:
        predicted_label = 'Dog'
    plt.title(f'Predicted: {predicted_label} (True: {true_label})')
    plt.show()