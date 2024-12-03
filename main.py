
from include import *
from data_augmentation import *

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
#model.add(Dense(1, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))


model.summary()

# Compile
# model.compile(loss = BinaryCrossentropy(),
#                 optimizer = keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.9),
#                 metrics = ['accuracy'])
model.compile(loss = keras.losses.CategoricalCrossentropy(),
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
test_dir_path = "pic-dataset/test"
# Defining Validation_generator withour Data Augmentation
data_test_gen = ImageDataGenerator(rescale = 1/255.)

print('Test Validation Images:')
test_ds = data_gen.flow_from_directory(test_dir_path, 
                                        target_size = (224, 224), 
                                        batch_size = 32,
                                        subset = 'validation',
                                        class_mode = 'categorical')

# test_ds = data_gen.flow_from_directory(test_dir_path, 
#                                         target_size = (224, 224), 
#                                         batch_size = 32,
#                                         subset = 'validation',
#                                         class_mode = 'binary')

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
image_paths = ['pic-dataset/train/cats/cat.0.jpg', 'pic-dataset/train/cats/cat.1.jpg', 'pic-dataset/train/dogs/dog.1.jpg', 'pic-dataset/train/foxes/2BWB37STRFS1.jpg']
# Intialize true labels
true_labels = ['Cat', 'Cat','Dog', 'Fox']

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
    #actual_prediction = (predictions > 0.5).astype(int)
    class_labels = ['Cat', 'Dog', 'Fox']  # Define el orden de las clases
    actual_prediction = np.argmax(predictions, axis=1)  # Índice de la clase con mayor probabilidad
    predicted_label = class_labels[actual_prediction[0]]

    # Display the image with true and predicted labels
    # Convert BGR to RGB for displaying with matplotlib
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  
    plt.axis('off')
    # if actual_prediction[0][0] == 0:
    #     predicted_label = 'Cat'
    # else:
    #     predicted_label = 'Dog'
    plt.title(f'Predicted: {predicted_label} (True: {true_label})')
    plt.show()