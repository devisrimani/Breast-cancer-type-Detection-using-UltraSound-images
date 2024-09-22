import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score
import random

import os
import cv2
from PIL import Image
import numpy as np

image_directory = r"C:\Users\DEVADHARSHINI K\Downloads\archive\Dataset_BUSI_with_GT"

def load_images(image_folder, label_value):
    images = [img for img in os.listdir(image_folder)]
    for image_name in images:
        if image_name.split('.')[1] == 'png' and '_mask' not in image_name:
            image = cv2.imread(os.path.join(image_folder, image_name))
            if image is not None:
                image = Image.fromarray(image, 'RGB')
                image = image.resize((SIZE, SIZE))
                image = np.array(image)
                dataset.append(image)
                label.append(label_value)

SIZE = 224
dataset = []
label = []

load_images(r'C:\Users\DEVADHARSHINI K\Downloads\archive\Dataset_BUSI_with_GT\benign', 0)  # Benign class with label 0
load_images(r'C:\Users\DEVADHARSHINI K\Downloads\archive\Dataset_BUSI_with_GT\malignant', 1)  # Malignant class with label 1
load_images(r'C:\Users\DEVADHARSHINI K\Downloads\archive\Dataset_BUSI_with_GT\normal', 2)  # Normal class with label 2

# Convert dataset and label to numpy arrays
dataset = np.array(dataset)
label = np.array(label)
print("Dataset shape:", dataset.shape)
print("Label shape:", label.shape)


from sklearn.model_selection import train_test_split

num_samples, height, width, channels = dataset.shape
X_flat = dataset.reshape(num_samples, -1)  # Reshape to (samples, height*width*channels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_flat, label, test_size=0.25, random_state=42)

#Applying different augmentation settings to minority classes:
augmentation_class1 = ImageDataGenerator(
    #rescale=1./255,
    rotation_range=5,  # Rotate images by a maximum of 10 degrees
    width_shift_range=0.1,  # Shift images horizontally by 10% of the width
    height_shift_range=0.1,  # Shift images vertically by 10% of the height
    zoom_range=0.1,  # Zoom images by 10%
    horizontal_flip=True,  # Flip images horizontally
    vertical_flip=False  # No vertical flipping
)
augmentation_class2 = ImageDataGenerator(
    rotation_range=30,  # Rotate images by a maximum of 10 degrees
    width_shift_range=0.2,  # Shift images horizontally by 10% of the width
    height_shift_range=0.2,  # Shift images vertically by 10% of the height
    zoom_range=0.2,  # Zoom images by 10%
    horizontal_flip=True,  # Flip images horizontally
    vertical_flip=True
)

X_train = X_train.reshape(-1, 224, 224, 3)  # Reshape your input data to match the expected input shape

datagen = ImageDataGenerator(
    horizontal_flip=True,   # Flip images horizontally
    vertical_flip=True,     # Flip images vertically
    fill_mode='nearest'     # Fill in missing pixels using the nearest available
)
datagen.fit(X_train)
augmented_images = []
augmented_labels = []

# Number of times to augment the data (in this case, we'll double the dataset)
augmentation_factor = 4

for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=len(X_train), shuffle=False):
    augmented_images.append(x_batch)
    augmented_labels.append(y_batch)
    if len(augmented_images) >= augmentation_factor:
        break

# Concatenate the augmented data batches
X_train = np.concatenate(augmented_images)
y_train = np.concatenate(augmented_labels)

# Verify the shape of augmented data
print("Shape of augmented images:", X_train.shape)
print("Shape of augmented labels:", y_train.shape)

def apply_augmentation(X_train, y_train):
    if y_train == 1:  # Check for class 1
        return augmentation_class1.random_transform(X_train), y_train
    if y_train == 2:
        return augmentation_class2.random_transform(X_train), y_train
    else:
        return X_train, y_train
X_test= X_test.reshape(-1, 224, 224, 3)  # Reshape your input data to match the expected input shape



from sklearn.utils.class_weight import compute_class_weight
class_labels = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=class_labels, y=y_train)
class_weight = {i: class_weights[i] for i in range(len(class_weights))}

base_model =EfficientNetV2B2(weights='imagenet', include_top=False,
                            input_shape=(224, 224, 3))

# Add custom top layers for your 3-class classification with regularization and dropout
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

# Freeze layers from the pre-trained model
for layer in base_model.layers[-20:]:  # Unfreeze last 20 layers for fine-tuning
    layer.trainable = True
    
    
# Compile the model with a lower learning rate and different optimizer
model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation and the modified architecture
history = model.fit(X_train,y_train,class_weight=class_weight, epochs=3, 
                    validation_data=(X_test,y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Predict classes for test data using model.predict
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Generate and print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Generate and print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

import matplotlib.pyplot as plt
# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Print best accuracy achieved during training
best_training_accuracy = max(history.history['accuracy'])
print("Best Training Accuracy: ", best_training_accuracy,"\n")

# Print approximate accuracy (validation accuracy)
approximate_accuracy = history.history['val_accuracy'][-1]
print("Approximate Accuracy (Validation Accuracy):", approximate_accuracy,"\n")

# Print top validation accuracy
top_validation_accuracy = max(history.history['val_accuracy'])
print("Top Validation Accuracy:", top_validation_accuracy,"\n")

from tensorflow.keras.models import load_model

# Assuming 'model' is your trained model
model.save('breast_model.h5')

import math
from tensorflow.keras.models import load_model
# Load the saved model
model = load_model('breast_model.h5')

def predict_and_show_all_images(model, dataset, true_labels, images_per_row=5):
    num_images = len(dataset)
    num_rows = math.ceil(num_images / images_per_row)
    
    for i in range(num_rows):
        fig, axes = plt.subplots(1, images_per_row, figsize=(20, 4))
        
        for j in range(images_per_row):
            index = i * images_per_row + j
            
            if index < num_images:
                # Expand dimensions to match model input shape
                image = np.expand_dims(dataset[index], axis=0)

                # Predict the class probabilities
                class_probabilities = model.predict(image)

                # Get the predicted class label
                predicted_class = np.argmax(class_probabilities)

                # Define class labels (you may need to adjust these according to your dataset)
                class_labels = ['benign', 'malignant', 'normal']

                # Display the image
                axes[j].imshow(image.squeeze())  # Squeeze to remove the batch dimension
                axes[j].axis('off')
                axes[j].set_title('True Label: {}\nPredicted Label: {}'.format(class_labels[true_labels[index]], class_labels[predicted_class]))
        
        plt.tight_layout()
        plt.show()
predict_and_show_all_images(model, dataset,label)

