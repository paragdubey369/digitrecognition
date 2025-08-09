import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # layer of nn
from tensorflow.keras.layers import Dense     # connected
from tensorflow.keras.activations import linear, relu    # activation function
import matplotlib.pyplot as plt 

def my_softmax(z):
    # Calculating the exponential of each element in z.
    ez = np.exp(z) # positive and magnify input
    # Sum all the exponential values. This will be the denominator for normalization.
    sum_ez = np.sum(ez)
    # Divide each exponential value by the sum to get the probabilities.
    a = ez / sum_ez
    return a
def display_digit(image_vector, actual_label, predicted_label=None):
    image_2d = image_vector.reshape(28, 28)

    plt.figure(figsize=(2, 2)) 
    plt.imshow(image_2d, cmap='gray') # Display as grayscale image
    plt.axis('off') 

    title = f"Actual: {actual_label}"
    if predicted_label is not None:
        title += f"\nPredicted: {predicted_label}"
    plt.title(title)
    plt.show() 
# - Neural Networks for Handwritten Digit Recognition ---

print("--- Loading MNIST Dataset ---")
#  the MNIST dataset. It returns tuples of (images, labels) for train and test sets.
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X = X_train.reshape(-1, 28 * 28).astype(np.float32)
y = y_train.reshape(-1, 1).astype(np.int32) # Labels are already 0-9 integers

# Original pixel values are typically 0-255.
X = X / 255.0
print(f"Loaded and preprocessed X shape: {X.shape}")
print(f"Loaded and preprocessed y shape: {y.shape}")
m, n = X.shape 
num_classes = 10 # Digits 0-9
print(f"\nFirst element of loaded X (first 10 pixels, normalized): {X[0,:10]}")
print(f"First element of loaded y: {y[0,0]}")

# 4.4 Tensorflow Model Implementation 
print("\n--- Building the TensorFlow Keras Model ---")
# Set a random seed for consistent results, especially for weight initialization.
tf.random.set_seed(1234)

# Define the Sequential model. Layers are added in order.
model = Sequential(
    [
        tf.keras.Input(shape=(n,)), # n = 784
        Dense(25, activation='relu', name="l1"),

        Dense(15, activation='relu', name="l2"),
        Dense(num_classes, activation='linear', name="l3") # num_classes = 10
    ],
    name="my_model"
)

# Display the model summary to see the layers, output shapes, and parameter counts.

model.summary()

[layer1, layer2, layer3] = model.layers

W1, b1 = layer1.get_weights()
W2, b2 = layer2.get_weights()
W3, b3 = layer3.get_weights()

print(f"\nWeights and Biases Shapes:")
print(f"W1 shape = {W1.shape}, b1 shape = {b1.shape}") 
print(f"W2 shape = {W2.shape}, b2 shape = {b2.shape}")
print(f"W3 shape = {W3.shape}, b3 shape = {b3.shape}")


print("\n--- Compiling the Model ---")
model.compile(
    # Loss function: SparseCategoricalCrossentropy is used for multiclass classification
 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
   
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    
    metrics=['accuracy'] # Adding accuracy to see classification performance
)

# --- Training the Model ---
print("\n--- Training the Model ---")

history = model.fit(
    X, y,
    epochs=10 
)
# --- Making Predictions ---
print("\n--- Making Predictions ---")

# Select a single random image from the (preprocessed) training set for prediction
random_index = np.random.randint(m)
image_for_prediction_input = X[random_index].reshape(1, n)  # Reshape 
actual_label = y[random_index, 0]

print(f"Display the image that was randomly selected for prediction:")
display_digit(X[random_index], actual_label)

prediction_logits = model.predict(image_for_prediction_input)

print(f"Raw prediction logits: \n{prediction_logits}")

# This converts the logits into a probability distribution over the 10 classes.
prediction_probabilities = tf.nn.softmax(prediction_logits).numpy() # .numpy() converts TensorFlow tensor to NumPy array

print(f"\nPrediction probabilities (softmax output): \n{prediction_probabilities}")
print(f"Sum of prediction probabilities: {np.sum(prediction_probabilities):0.3f}")

# To determine the predicted digit, find the index of the highest probability.
predicted_digit = np.argmax(prediction_probabilities)
print(f"\nPredicted digit (index with highest probability): {predicted_digit}")

# --- Display the image again,  with the prediction ---
print(f"\nDisplaying the image with its actual and predicted label:")
display_digit(X[random_index], actual_label, predicted_digit)
