import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Directory setup
base_dir = "nether_images"
mushroom_dir = os.path.join(base_dir, "mushroom")
no_mushroom_dir = os.path.join(base_dir, "no_mushroom")
output_dir = "boxed_shrooms"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def load_and_preprocess_images(directory, label):
    images = []
    labels = []
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            img = img / 255.0  # Normalize pixel values
            images.append(img)
            labels.append(label)
    return images, labels

# Load images
mushroom_images, mushroom_labels = load_and_preprocess_images(mushroom_dir, 1)
no_mushroom_images, no_mushroom_labels = load_and_preprocess_images(no_mushroom_dir, 0)

# Combine datasets
X = np.array(mushroom_images + no_mushroom_images)
y = np.array(mushroom_labels + no_mushroom_labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, 
                    epochs=10,
                    validation_data=(X_test, y_test))

# Evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

# Function to detect and draw boxes around mushrooms
def detect_mushrooms(image):
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Red mushroom mask
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Brown mushroom mask
    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([20, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Find contours
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    brown_contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw boxes
    for contour in red_contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    for contour in brown_contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return image

# Process and save images with boxes
for img_name in os.listdir(mushroom_dir):
    img_path = os.path.join(mushroom_dir, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        boxed_img = detect_mushrooms(img.copy())
        cv2.imwrite(os.path.join(output_dir, f'boxed_{img_name}'), boxed_img)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
